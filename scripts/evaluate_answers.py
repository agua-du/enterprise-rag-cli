from __future__ import annotations

import json
import runpy
import subprocess
import sys
from http import HTTPStatus
from pathlib import Path

import dashscope
from dashscope import MultiModalConversation

from enterprise_rag_cli.application.benchmark import write_json
from enterprise_rag_cli.config import get_settings
from enterprise_rag_cli.infrastructure.benchmark import iter_jsonl
from enterprise_rag_cli.infrastructure.documents import file_sha256


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools" / "EnterpriseRAG-Bench"
HARNESS_COMMIT = "d36685e273713975ee20299bbf1ab64165575b3c"

# 裁判模型调用
class QwenJudge:
    def __init__(self, settings):
        self.settings = settings
        self.api_errors = 0

    def generate(self, messages):
        try:
            response = MultiModalConversation.call(
                api_key=self.settings.dashscope_api_key.get_secret_value().strip(),
                model=self.settings.judge_model,
                messages=[
                    {"role": item.role, "content": [{"text": item.content}]}
                    for item in messages
                ],
                result_format="message",
                enable_thinking=False,
                temperature=0,
            )
        except Exception:
            self.api_errors += 1
            raise
        if response.status_code != HTTPStatus.OK:
            self.api_errors += 1
            raise RuntimeError(
                f"裁判调用失败：{response.code} {response.message}"
            )
        text = "".join(
            item.get("text", "")
            for item in response.output.choices[0].message.content
        ).strip()
        if not text:
            self.api_errors += 1
            raise RuntimeError("裁判模型返回了空内容")
        yield text


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if not manifest["completed"] or manifest["errors"]:
        raise RuntimeError("问答运行未完整成功，请先检查 traces.jsonl")
    questions_path = run_dir / "questions.jsonl"
    if file_sha256(questions_path) != manifest["questions_sha256"]:
        raise ValueError("本次运行的问题快照已变化")
    expected_ids = {row["question_id"] for row in iter_jsonl(questions_path)}
    answers = list(iter_jsonl(run_dir / "answers.jsonl"))
    # 检查回答有没有缺失、重复或为空
    if (
        len(answers) != len(expected_ids)
        or {row["question_id"] for row in answers} != expected_ids
        or any(not row.get("answer") for row in answers)
    ):
        raise ValueError("回答文件缺题、重复或包含空答案")

    # 检查官方评测器版本
    commit = subprocess.check_output(
        ["git", "-C", str(HARNESS), "rev-parse", "HEAD"], text=True,
    ).strip()
    if commit != HARNESS_COMMIT:
        raise ValueError("官方评测器版本与本章约定不一致")

    settings = get_settings()
    if (
        settings.dashscope_api_key is None
        or not settings.dashscope_api_key.get_secret_value().strip()
    ):
        raise RuntimeError("请先配置 RAG_DASHSCOPE_API_KEY")
    dashscope.base_http_api_url = str(
        settings.dashscope_base_http_api_url
    ).rstrip("/")
    judge = QwenJudge(settings)

    # 替换官方评测器的模型入口
    sys.path.insert(0, str(HARNESS))
    # 把评测器目录放到 Python 模块搜索路径的最前面，使下面这些导入能找到评测器里的 src 包
    import src.llm as official_llm
    import src.llm.factory as factory

    # 定义一个替代的模型创建函数
    def create_judge(*args, **kwargs):
        return judge

    official_llm.get_llm = create_judge
    official_llm.get_cheap_llm = create_judge
    factory.get_llm = create_judge
    factory.get_cheap_llm = create_judge

    # 写入本次评分的元信息
    judge_info = {
        "model": settings.judge_model,
        "base_url": str(settings.dashscope_base_http_api_url),
        "harness_commit": commit,
        "adapter_sha256": file_sha256(Path(__file__)),
        "no_correction": True,
        "completed": False,
    }
    write_json(run_dir / "judge.json", judge_info)
    # 在当前进程里运行官方评分脚本
    sys.argv = [
        "metrics_based_eval",
        "--questions-file", str(questions_path),
        "--answers-file", str(run_dir / "answers.jsonl"),
        "--results-file", str(run_dir / "scores.json"),
        "--no-correction",
        "--parallelism", "1",
    ]
    runpy.run_module(
        "src.scripts.answer_evaluation.metrics_based_eval", run_name="__main__",
    )
    # 记录最终状态
    judge_info["api_errors"] = judge.api_errors
    judge_info["completed"] = judge.api_errors == 0
    write_json(run_dir / "judge.json", judge_info)
    if judge.api_errors:
        raise RuntimeError("裁判接口出现错误，本次评分不能用于比较，请排查后重新评分")


if __name__ == "__main__":
    main()