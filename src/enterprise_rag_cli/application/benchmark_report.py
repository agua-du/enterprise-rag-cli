from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from enterprise_rag_cli.infrastructure.benchmark import iter_jsonl
from enterprise_rag_cli.infrastructure.documents import file_sha256


def index_rows(rows: list[dict]) -> dict[str, dict]:
    indexed = {row["question_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("结果文件中存在重复的问题 ID")
    return indexed


def load_evaluated_run(run_dir: Path) -> tuple[dict, dict, list[dict]]:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    judge = json.loads((run_dir / "judge.json").read_text(encoding="utf-8"))
    if not manifest["completed"] or manifest["errors"]:
        raise ValueError("问答运行存在失败，请先检查 traces.jsonl")
    if not judge["completed"]:
        raise ValueError("裁判评分未完整完成")

    questions_path = run_dir / "questions.jsonl"
    if file_sha256(questions_path) != manifest["questions_sha256"]:
        raise ValueError("问题快照已变化")
    questions = index_rows(list(iter_jsonl(questions_path)))
    answers = index_rows(list(iter_jsonl(run_dir / "answers.jsonl")))
    traces = index_rows(list(iter_jsonl(run_dir / "traces.jsonl")))
    scores_file = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    scores = index_rows(scores_file["questions"])
    if not questions or any(
        set(rows) != set(questions) for rows in (answers, traces, scores)
    ):
        raise ValueError("问题、回答、证据和评分没有完整对齐")

    rows = []
    for question_id, question in questions.items():
        trace = traces[question_id]
        score = scores[question_id]
        if trace["status"] != "ok" or trace["retrieval"] is None:
            raise ValueError(f"{question_id} 缺少正常的检索记录")
        rows.append({
            "question_id": question_id,
            "question_type": question["question_type"],
            "source_types": question["source_types"],
            "question": question["question"],
            "gold_answer": question["gold_answer"],
            "answer_facts": question["answer_facts"],
            "expected_doc_ids": question["expected_doc_ids"],
            "answer": answers[question_id]["answer"],
            "document_ids": answers[question_id]["document_ids"],
            "evidence": trace["evidence"],
            "latency_seconds": trace["latency_seconds"],
            **trace["retrieval"],
            "answer_correct": score["answer_correct"],
            "completeness_pct": score["completeness_pct"],
            "invalid_extra_docs": score["invalid_extra_docs"],
            "correctness_reasoning": score["correctness_reasoning"],
        })
    return manifest, judge, rows


def summarize_results(rows: list[dict]) -> pl.DataFrame:
    fields = [
        "question_type", "answer_correct", "completeness_pct",
        "document_recall_pct", "hit", "reciprocal_rank", "latency_seconds",
    ]
    frame = pl.DataFrame([{key: row[key] for key in fields} for row in rows])
    # 同时生成整体行和按问题类型分组的行。
    frame = pl.concat([
        frame,
        frame.with_columns(pl.lit("全部").alias("question_type")),
    ])
    return (
        frame.group_by("question_type")
        .agg(
            pl.len().alias("questions"),
            (pl.col("answer_correct").mean() * 100).alias("correctness_pct"),
            pl.col("completeness_pct").mean(),
            pl.col("document_recall_pct").mean(),
            (pl.col("hit").mean() * 100).alias("hit_rate_pct"),
            pl.col("reciprocal_rank").mean().alias("mrr"),
            pl.col("latency_seconds").mean(),
        )
        .sort("question_type")
    )


def save_review_cases(rows: list[dict], path: Path) -> int:
    review = [
        row for row in rows
        if not row["answer_correct"]
        or row["completeness_pct"] < 100
        or row["document_recall_pct"] < 100
    ]
    with path.open("w", encoding="utf-8") as file:
        for row in review:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(review)

def compare_benchmarks(baseline_dir: Path, candidate_dir: Path) -> list[dict]:
    baseline, baseline_judge, old_rows = load_evaluated_run(baseline_dir)
    candidate, candidate_judge, new_rows = load_evaluated_run(candidate_dir)
    for key in ("questions_sha256", "corpus_sha256"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"两次实验的 {key} 不一致，不能直接比较")
    for key in ("model", "base_url", "harness_commit", "adapter_sha256", "no_correction"):
        if baseline_judge[key] != candidate_judge[key]:
            raise ValueError(f"两次实验的裁判配置 {key} 不一致")

    old = index_rows(old_rows)
    new = index_rows(new_rows)
    if set(old) != set(new):
        raise ValueError("两次实验的问题 ID 没有完整对齐")

    comparison = []
    for question_id in sorted(old):
        before, after = old[question_id], new[question_id]
        comparison.append({
            "question_id": question_id,
            "question_type": before["question_type"],
            "baseline_correct": before["answer_correct"],
            "candidate_correct": after["answer_correct"],
            "correctness_delta": int(after["answer_correct"]) - int(before["answer_correct"]),
            "completeness_delta": after["completeness_pct"] - before["completeness_pct"],
            "recall_delta": after["document_recall_pct"] - before["document_recall_pct"],
            "latency_delta": after["latency_seconds"] - before["latency_seconds"],
        })
    return comparison
