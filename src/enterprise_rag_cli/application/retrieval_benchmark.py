from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import polars as pl

from enterprise_rag_cli.application.benchmark import (
    code_sha256,
    retrieval_metrics,
    write_json,
)
from enterprise_rag_cli.config import Settings
from enterprise_rag_cli.infrastructure.benchmark import iter_jsonl
from enterprise_rag_cli.infrastructure.documents import file_sha256
from enterprise_rag_cli.infrastructure.embeddings import embed_query
from enterprise_rag_cli.infrastructure.vector_store import (
    create_milvus_client,
    search_chunks,
)

# 测试不同ef参数下检索质量和延迟
def run_retrieval_benchmark(
    settings: Settings,
    suite_dir: Path,
    output_dir: Path,
    top_k: int,
    ef_values: list[int],
) -> list[dict]:
    ef_values = sorted(set(ef_values))
    if not ef_values or not 1 <= top_k <= min(ef_values):
        raise ValueError("每个 ef 都需要不小于 top_k，且 top_k 必须大于 0")
    if max(ef_values) > 32768:
        raise ValueError("ef 不能超过 32768")

    suite = json.loads((suite_dir / "suite.json").read_text(encoding="utf-8"))
    questions_path = suite_dir / "questions.jsonl"
    if file_sha256(questions_path) != suite["questions_sha256"]:
        raise ValueError("评测问题文件已变化，请重新准备评测集")
    questions = list(iter_jsonl(questions_path))
    if not questions or len(questions) != suite["question_count"]:
        raise ValueError("问题数量与评测集记录不一致")

    # 使用评测集记录的实体集合，避免 Alias 切换影响本次实验。
    collection = suite["collection"]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "questions.jsonl").write_bytes(questions_path.read_bytes())
    manifest = {
        **suite,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "top_k": top_k,
        "ef_values": ef_values,
        "code_sha256": code_sha256(),
        "completed": False,
    }
    write_json(output_dir / "run.json", manifest)

    rows = []
    client = create_milvus_client(settings)
    try:
        client.load_collection(collection_name=collection)
        manifest["indexes"] = [
            client.describe_index(collection_name=collection, index_name=name)
            for name in client.list_indexes(collection_name=collection)
        ]
        write_json(output_dir / "run.json", manifest)

        with (output_dir / "retrieval.jsonl").open("w", encoding="utf-8") as file:
            for number, question in enumerate(questions):
                started = perf_counter()
                vector = embed_query(question["question"], settings)
                embedding_ms = (perf_counter() - started) * 1000

                # 第一题先预热各组搜索，预热结果不计入统计。
                if number == 0:
                    for ef in ef_values:
                        search_chunks(client, collection, vector, top_k, ef)

                # 轮换各组的执行顺序，减少固定先后次序的影响。
                offset = number % len(ef_values)
                order = ef_values[offset:] + ef_values[:offset]
                for ef in order:
                    started = perf_counter()
                    evidence = search_chunks(
                        client, collection, vector, top_k, ef,
                    )
                    search_ms = (perf_counter() - started) * 1000
                    metrics = retrieval_metrics(
                        question["expected_doc_ids"], evidence,
                    )
                    row = {
                        "question_id": question["question_id"],
                        "question_type": question["question_type"],
                        "ef": ef,
                        "embedding_ms": round(embedding_ms, 3),
                        "search_ms": round(search_ms, 3),
                        "retrieval": metrics,
                        "evidence": [item.model_dump(mode="json") for item in evidence],
                    }
                    rows.append(row)
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    file.flush()
    finally:
        client.close()

    frame = pl.DataFrame([
        {
            "ef": row["ef"],
            "question_type": row["question_type"],
            "search_ms": row["search_ms"],
            **row["retrieval"],
        }
        for row in rows
    ])
    summary = (
        pl.concat([
            frame,
            frame.with_columns(pl.lit("全部").alias("question_type")),
        ])
        .group_by("ef", "question_type")
        .agg(
            pl.len().alias("questions"),
            pl.col("document_recall_pct").mean(),
            (pl.col("hit").mean() * 100).alias("hit_rate_pct"),
            pl.col("reciprocal_rank").mean().alias("mrr"),
            pl.col("search_ms").mean(),
        )
        .sort("question_type", "ef")
    )
    summary.write_csv(output_dir / "summary.csv")
    manifest["completed"] = True
    write_json(output_dir / "run.json", manifest)
    return summary.to_dicts()