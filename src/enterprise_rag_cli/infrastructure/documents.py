from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from enterprise_rag_cli.domain.ids import make_document_id
from enterprise_rag_cli.domain.models import Document, SourceDocument


SOURCE_ID_PATTERN = re.compile(r"^(dsid_[0-9a-f]+)")

#文件 hash
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

# 从文件名中提取 source_id
def extract_source_id(path: Path) -> str | None:
    match = SOURCE_ID_PATTERN.match(path.name)
    return match.group(1) if match else None

# 寻找 document 目录下的所有 txt，根据 souce_type 和limit来过滤
def discover_source_documents(
    root: Path,
    source_type: str | None = None,
    limit: int | None = None,
) -> Iterator[SourceDocument]:
    if limit is not None and limit < 1:
        raise ValueError("limit 必须大于 0")

    discovered = 0

    for path in sorted(root.rglob("*.txt")):
        relative_path = path.relative_to(root)
        if len(relative_path.parts) < 2:
            continue

        current_source_type = relative_path.parts[0]
        if source_type is not None and current_source_type != source_type:
            continue

        source_id = extract_source_id(path)
        if source_id is None:
            continue

        yield SourceDocument(
            source_type=current_source_type,
            source_id=source_id,
            path=path,
            content_sha256=file_sha256(path),
        )

        discovered += 1
        if limit is not None and discovered >= limit:
            return

# 读取文档，返回Document
def parse_text_document(source: SourceDocument) -> Document:
    text = source.path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"文档内容为空：{source.path}")

    title = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        None,
    )

    return Document(
        document_id=make_document_id(source.source_type, source.source_id),
        source_type=source.source_type,
        source_id=source.source_id,
        title=title,
        text=text,
        content_sha256=source.content_sha256,
    )