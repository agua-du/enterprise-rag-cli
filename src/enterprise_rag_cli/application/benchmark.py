from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from time import perf_counter

from enterprise_rag_cli.config import Settings
from enterprise_rag_cli.infrastructure.benchmark import iter_jsonl
from enterprise_rag_cli.infrastructure.documents import file_sha256
from enterprise_rag_cli.infrastructure.vector_store import (
    collect_index_document_ids,
)
from enterprise_rag_cli.application.rag import answer_question
from enterprise_rag_cli.domain.models import RetrievalResult

# 把数据写入文件中，保存为 json
def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

# benchmark 准备工作
def prepare_benchmark(settings: Settings, output_dir: Path) -> dict:
    collection, document_ids = collect_index_document_ids(settings)
    questions = []
    seen_ids: set[str] = set()

    for question in iter_jsonl(settings.data_dir / "questions.jsonl"):
        question_id = question["question_id"]
        # 有重复的，有问题
        if question_id in seen_ids:
            raise ValueError(f"问题 ID 重复：{question_id}")
        seen_ids.add(question_id)

        # high_level和info_not_found的类型，我们先不做
        if question["question_type"] in {"high_level", "info_not_found"}:
            continue
        expected = set(question.get("expected_doc_ids", []))
        # <=不是数值大小比较，是 Python 集合的子集判断，这里需要判断 expected 是否为空集合，因为空集合是任何集合的子集
        if expected and expected <= document_ids:
            questions.append(question)

    if not questions:
        raise ValueError("当前索引没有覆盖任何预期文档齐全的问题")

    #根据question_id排序
    questions.sort(key=lambda item: item["question_id"])

    # 记录索引集合、文档数量、问题数量，以及文档 ID 清单和问题文件的指纹
    output_dir.mkdir(parents=True, exist_ok=False)
    questions_path = output_dir / "questions.jsonl"
    with questions_path.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(json.dumps(question, ensure_ascii=False) + "\n")

    corpus_text = "\n".join(sorted(document_ids))
    suite = {
        "collection": collection,
        "document_count": len(document_ids),
        "corpus_sha256": hashlib.sha256(corpus_text.encode()).hexdigest(),
        "question_count": len(questions),
        "questions_sha256": file_sha256(questions_path),
    }
    write_json(output_dir / "suite.json", suite)
    return suite

# 计算 文档召回率 ｜ 是否命中 ｜ 首次命中的倒数
def retrieval_metrics(
    expected_doc_ids: list[str],
    evidence: list[RetrievalResult],
) -> dict:
    expected = set(expected_doc_ids)
    if not expected:
        raise ValueError("本评测集要求问题包含预期文档")

    # 召回文档集合
    retrieved = {item.source_id for item in evidence}
    # 返回第一个来自预期文档的 Chunk 的排名
    first_rank = next(
        (item.rank for item in evidence if item.source_id in expected),
        None,
    )
    return {
        "document_recall_pct": len(expected & retrieved) / len(expected) * 100,
        "hit": int(first_rank is not None),
        "reciprocal_rank": 1 / first_rank if first_rank is not None else 0.0,
    }

# 把当前包中的 Python 代码和 Prompt 一起计入指纹。
def code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".jinja2"}:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()

# 运行 benchmark
def run_benchmark(
    settings: Settings,
    suite_dir: Path,
    output_dir: Path,
    top_k: int,
) -> dict:
    # 获取过滤后的question.jsonl中问题
    suite = json.loads((suite_dir / "suite.json").read_text(encoding="utf-8"))
    questions_path = suite_dir / "questions.jsonl"
    if file_sha256(questions_path) != suite["questions_sha256"]:
        raise ValueError("评测问题文件已变化，请重新准备评测集")
    questions = list(iter_jsonl(questions_path))
    if len(questions) != suite["question_count"]:
        raise ValueError("问题数量与评测集记录不一致")

    # 复制 Pydantic 配置对象,使用准备评测集时记录的Collection。
    run_settings = settings.model_copy(
        update={"milvus_collection": suite["collection"]},
    )
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=False)
    # 拷贝一份questions.jsonl
    (output_dir / "questions.jsonl").write_bytes(questions_path.read_bytes())
    manifest = {
        **suite,
        "top_k": top_k,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "generation_model": settings.generation_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "code_sha256": code_sha256(),
        "completed": False,
        "errors": 0,
    }
    # manifest写入run.json
    write_json(output_dir / "run.json", manifest)

    with (
        (output_dir / "answers.jsonl").open("w", encoding="utf-8") as answers,
        (output_dir / "traces.jsonl").open("w", encoding="utf-8") as traces,
    ):
        for question in questions:
            # 开始时间，用来计算运行时间
            started = perf_counter()
            answer_row = {
                "question_id": question["question_id"],
                "answer": "",
                "document_ids": [],
            }
            trace = {
                "question_id": question["question_id"],
                "status": "ok",
                "evidence": [],
                "retrieval": None,
                "error": None,
            }
            try:
                # 回答问题
                result = answer_question(
                    question=question["question"],
                    settings=run_settings,
                    top_k=top_k,
                )
                answer_row["answer"] = result.answer
                # dict.fromkeys 按顺序去重
                answer_row["document_ids"] = list(dict.fromkeys(
                    item.source_id for item in result.evidence
                ))
                trace["evidence"] = [
                    item.model_dump(mode="json") for item in result.evidence
                ]
                trace["retrieval"] = retrieval_metrics(
                    question["expected_doc_ids"], result.evidence,
                )
            except Exception as error:
                trace["status"] = "error"
                trace["error"] = f"{type(error).__name__}: {error}"
                manifest["errors"] += 1

            trace["latency_seconds"] = round(perf_counter() - started, 3)
            # 写入文件并刷新文件缓冲区
            answers.write(json.dumps(answer_row, ensure_ascii=False) + "\n")
            traces.write(json.dumps(trace, ensure_ascii=False) + "\n")
            answers.flush()
            traces.flush()

    manifest["completed"] = True
    write_json(output_dir / "run.json", manifest)
    return manifest