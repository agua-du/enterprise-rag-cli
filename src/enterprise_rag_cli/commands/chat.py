from __future__ import annotations

import typer
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from enterprise_rag_cli.application.rag import answer_question
from enterprise_rag_cli.config import get_settings
from enterprise_rag_cli.presentation import console


def chat(
    question: str = typer.Argument(..., help="要询问的问题"),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        max=20,
        help="返回的检索结果数量",
    ),
) -> None:
    """根据当前文档库回答一个问题。"""

    settings = get_settings()
    actual_top_k = top_k or settings.default_top_k

    with console.status("正在检索文档并生成答案…"):
        result = answer_question(
            question=question,
            settings=settings,
            top_k=actual_top_k,
        )

    console.print(
        Panel(
            Markdown(result.answer),
            title="回答",
            border_style="green",
        )
    )

    table = Table(title="检索来源")
    table.add_column("编号", justify="right")
    table.add_column("相似度", justify="right")
    table.add_column("文档来源")
    table.add_column("内容预览")

    for item in result.evidence:
        preview = " ".join(item.text.split())[:120]
        table.add_row(
            str(item.rank),
            f"{item.score:.4f}",
            f"{item.source_type}/{item.source_id}",
            preview,
        )

    console.print(table)