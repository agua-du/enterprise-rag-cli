from __future__ import annotations

import json
from pathlib import Path

from enterprise_rag_cli.application.chunking import (
    create_chunker, meeting_rules, split_document,
)
from enterprise_rag_cli.domain.models import Chunk, ChunkPolicy, Document

# 读取分割策略配置
def load_chunk_policies(path: Path) -> dict[str, ChunkPolicy]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "default" not in data:
        raise ValueError("切分配置必须包含 default 规则")
    return {
        source: ChunkPolicy.model_validate(value)
        for source, value in data.items()
    }

# 根据策略分割文档
def split_with_policy(document: Document, policy: ChunkPolicy) -> list[Chunk]:
    rules = meeting_rules() if policy.structure == "meeting" else None
    chunker = create_chunker(
        policy.chunk_size, policy.chunk_overlap, policy.strategy, rules=rules,
    )
    return split_document(document, chunker, include_title=policy.include_title)