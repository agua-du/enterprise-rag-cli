from __future__ import annotations

import typer
from rich.panel import Panel
from rich.text import Text
from enum import Enum

from enterprise_rag_cli.application.chunking import create_chunker, split_document
from enterprise_rag_cli.config import get_settings
from enterprise_rag_cli.infrastructure.database import create_session_factory
from enterprise_rag_cli.infrastructure.document_repository import load_parsed_document
from enterprise_rag_cli.presentation import console
from enterprise_rag_cli.application.chunk_policies import load_chunk_policies, split_with_policy
from enterprise_rag_cli.domain.models import ChunkPolicy

class ChunkStrategy(str, Enum):
    token = "token"
    sentence = "sentence"
    recursive = "recursive"


app = typer.Typer(help="查看文档切分结果", no_args_is_help=True)


@app.command("preview")
def preview(
    source_id: str = typer.Argument(..., help="原始文档 ID，即 dsid_..."),
    source: str = typer.Option("fireflies", "--source", help="文档来源"),
    size: int | None = typer.Option(None, "--size", min=200),
    overlap: int | None = typer.Option(None, "--overlap", min=0),
    contains: str | None = typer.Option(None, "--contains", help="只显示包含指定文字的片段"),
    limit: int = typer.Option(3, "--limit", min=1, max=20),
    strategy: ChunkStrategy | None = typer.Option(None, "--strategy"),
) -> None:
    """读取一篇文档并预览 Chunk。"""
    settings = get_settings()
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        document = load_parsed_document(session, source, source_id)

    # 分割
    policies = load_chunk_policies(settings.chunk_policies_path)
    policy = policies.get(document.source_type, policies["default"])
    overrides = {}
    if size is not None:
        overrides["chunk_size"] = size
    if overlap is not None:
        overrides["chunk_overlap"] = overlap
    if strategy is not None:
        overrides["strategy"] = strategy.value
        overrides["structure"] = "paragraph"
    policy = ChunkPolicy.model_validate({**policy.model_dump(), **overrides})
    chunks = split_with_policy(document, policy)
    console.print(f"切分规则：{policy.model_dump()}")

    selected = [
        chunk for chunk in chunks
        if contains is None or contains.casefold() in chunk.text.casefold()
    ]
    console.print(f"文档：{document.title or source_id}")
    console.print(f"原文字符：{len(document.text)}；Chunk 数量：{len(chunks)}")
    console.print(f"匹配片段：{len(selected)}；本次最多显示：{limit}")
    for chunk in selected[:limit]:
        console.print(Panel(
            Text(chunk.text),
            title=f"Chunk {chunk.position} · {len(chunk.text)} 字符",
        ))