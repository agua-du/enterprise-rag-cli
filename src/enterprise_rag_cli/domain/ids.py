from __future__ import annotations

import hashlib
from uuid import NAMESPACE_URL, uuid5


def make_document_id(source_type: str, source_id: str) -> str:
    source_type = source_type.strip()
    source_id = source_id.strip()
    if not source_type or not source_id:
        raise ValueError("source_type 和 source_id 不能为空")

    value = uuid5(NAMESPACE_URL, f"enterprise-rag:{source_type}:{source_id}")
    return str(value)


def make_chunk_id(document_id: str, text: str, position: int) -> str:
    if not text.strip():
        raise ValueError("分块正文不能为空")
    if position < 0:
        raise ValueError("position 不能小于 0")

    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = uuid5(NAMESPACE_URL, f"{document_id}:{position}:{text_sha256}")
    return str(value)