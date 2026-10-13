from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
# 处理数据的库，类似 sql
import polars as pl


DOCUMENT_ID_PATTERN = re.compile(r"^(dsid_[0-9a-f]+)")

# 文件hash
def file_sha256(path:Path)->str:
    digest = hashlib.sha256()
    with open(path,"rb") as file:
        while True:
            data = file.read(1024*1024)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()

# 读取questions.jsonl返回每行的 json
def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue

            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是有效 JSON"
                ) from error

            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")

            yield value


def load_questions(path: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in iter_jsonl(path):
        required_fields = {
            "question_id",
            "question_type",
            "source_types",
            "question",
            "gold_answer",
            "answer_facts",
        }
        missing = required_fields - item.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{item.get('question_id', '<unknown>')} 缺少字段: {names}")

        expected_doc_ids = item.get("expected_doc_ids", [])
        rows.append(
            {
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "source_type_count": len(item["source_types"]),
                "question_chars": len(item["question"]),
                "expected_doc_count": len(expected_doc_ids),
                "answer_fact_count": len(item["answer_facts"]),
                "expected_doc_ids": expected_doc_ids,
            }
        )

    if not rows:
        raise ValueError(f"{path} 中没有问题")

    return pl.DataFrame(rows)

#使用正则截取文件名中 dsid_[0-9a-f]这段作为文档id
def extract_document_id(path: Path) -> str | None:
    match = DOCUMENT_ID_PATTERN.match(path.name)
    return match.group(1) if match else None

# 扫描文档目录，生成字典[文档 id,文件问题类型，文件大小字节数]
def scan_documents(root: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in root.rglob("*.txt"):
        #documents 目录下面文件夹的名称为source_type
        relative = path.relative_to(root)
        source_type = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        document_id = extract_document_id(path)

        rows.append(
            {
                "document_id": document_id,
                "source_type": source_type,
                "bytes": path.stat().st_size,
            }
        )

    if not rows:
        raise ValueError(f"{root} 中没有 .txt 文档")

    return pl.DataFrame(rows)

#问题评测集 的数据统计
def summarize_questions(questions: pl.DataFrame) -> pl.DataFrame:
    return (
        #按question_type分组
        questions.group_by("question_type")
        #聚合数据
        .agg(
            # 每个question_type类型的问题测评数量，并命名为questions
            pl.len().alias("questions"),
            # 取expected_doc_count字段的平均值，保留 2 位小数点，并命名为avg_expected
            pl.col("expected_doc_count").mean().round(2).alias("avg_expected"),
            pl.col("answer_fact_count").mean().round(2).alias("avg_answer_facts"),
            # 取question_chars字段的中位数，保留 2 位小数点，并命名为median_question_chars
            pl.col("question_chars").median().alias("median_question_chars"),
        )
        #按question_type字段来排序
        .sort("question_type")
    )

# 文档统计
def summarize_documents(documents: pl.DataFrame) -> pl.DataFrame:
    return (
        documents.group_by("source_type")
        .agg(
            pl.len().alias("documents"),
            #每个source_type类型的文档的总大小(单位为 M)，命名为total_mb
            (pl.col("bytes").sum() / 1024 / 1024).round(2).alias("total_mb"),
            #每个source_type类型的文档大小的中位数，命名为median_bytes
            pl.col("bytes").median().cast(pl.Int64).alias("median_bytes"),
            #每个source_type类型的文档的95% 分位数值，命名为p95_bytes
            pl.col("bytes").quantile(0.95).cast(pl.Int64).alias("p95_bytes"),
        )
        #按documents数量降序排序
        .sort("documents", descending=True)
    )

#返回所有expected_doc_ids的集合
def expected_document_ids(questions: pl.DataFrame) -> set[str]:
    values = questions.get_column("expected_doc_ids").to_list()
    return {
        document_id
        for group in values
        for document_id in group
    }