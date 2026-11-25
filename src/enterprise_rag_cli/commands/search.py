from __future__ import annotations

import typer
from rich.panel import Panel
from rich.text import Text

from enterprise_rag_cli.application.retrieval import retrieve_chunks
from enterprise_rag_cli.config import get_settings
from enterprise_rag_cli.presentation import console


app = typer.Typer(help="查看检索结果", no_args_is_help=True)


@app.command("dense")
def dense(
    question: str = typer.Argument(..., help="需要检索的问题"),
    top_k: int = typer.Option(5, "--top-k", min=1, max=20),
    ef: int | None = typer.Option(
        None, "--ef", min=1, max=32768,
        help="HNSW 搜索范围，默认读取配置",
    ),
) -> None:
    """使用问题向量查找相关片段。"""
    settings = get_settings()
    actual_ef = settings.hnsw_ef_search if ef is None else ef
    if actual_ef < top_k:
        raise typer.BadParameter("ef 不能小于 top-k")

    with console.status("正在检索相关片段…"):
        results = retrieve_chunks(question, settings, top_k, actual_ef)

    console.print(f"返回片段：{len(results)}；ef：{actual_ef}")
    for item in results:
        title = (
            f"第 {item.rank} 条 · 相似度 {item.score:.4f}"
            f" · {item.source_type}/{item.source_id}"
            f" · Chunk 位置 {item.position}"
        )
        console.print(Panel(Text(item.text), title=Text(title)))