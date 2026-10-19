from __future__ import annotations

from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from sqlalchemy.exc import SQLAlchemyError

from enterprise_rag_cli.application.ingest import (
    IngestStats,
    ingest_documents,
)
from enterprise_rag_cli.config import get_settings
from enterprise_rag_cli.infrastructure.database import create_session_factory
from enterprise_rag_cli.infrastructure.document_repository import (
    ImportSummary,
    summarize_imports,
)
from enterprise_rag_cli.infrastructure.documents import (
    discover_source_documents,
)
from enterprise_rag_cli.presentation import console


def render_run_stats(stats: IngestStats) -> None:
    table = Table(title="本次导入")
    table.add_column("结果")
    table.add_column("文档数", justify="right")

    table.add_row("发现", str(stats.discovered))
    table.add_row("创建", str(stats.created))
    table.add_row("更新", str(stats.updated))
    table.add_row("未变化", str(stats.unchanged))
    table.add_row("失败", str(stats.failed))
    console.print(table)


def render_quality_summary(rows: list[ImportSummary]) -> None:
    table = Table(title="文档库导入状态")
    table.add_column("source_type")
    table.add_column("parsed", justify="right")
    table.add_column("failed", justify="right")

    for row in rows:
        table.add_row(
            row.source_type,
            str(row.parsed),
            str(row.failed),
        )

    console.print(table)


def ingest(
    documents_dir: Path | None = typer.Option(
        None,
        "--documents",
        help="文档库根目录；默认使用配置中的 data_dir/documents",
    ),
    source_type: str | None = typer.Option(
        None,
        "--source",
        help="只导入指定来源，例如 fireflies",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="最多导入多少份文档，适合先做小规模检查",
    ),
) -> None:
    """读取本地文档并保存到 PostgreSQL。"""

    settings = get_settings()
    if documents_dir is None:
        documents_dir = settings.data_dir / "documents"

    if not documents_dir.is_dir():
        console.print(f"[red]找不到文档库目录：{documents_dir}[/red]")
        raise typer.Exit(code=2)

    if source_type is not None:
        source_type = source_type.strip()
        if not source_type:
            console.print("[red]--source 不能为空[/red]")
            raise typer.Exit(code=2)

    with console.status("正在扫描文档并计算内容指纹…"):
        sources = list(
            discover_source_documents(
                root=documents_dir,
                source_type=source_type,
                limit=limit,
            )
        )

    if not sources:
        console.print("[yellow]没有找到符合条件的 .txt 文档[/yellow]")
        return

    session_factory = create_session_factory(settings)

    try:
        with session_factory() as session:
            # 显示进度条
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                # progress.track()返回一个可迭代对象，当代码逐个消费其中的元素时，Rich 会自动推进进度条
                tracked_sources = progress.track(
                    sources,
                    description="正在导入文档",
                )
                stats = ingest_documents(
                    session=session,
                    sources=tracked_sources,
                    documents_dir=documents_dir,
                )

            summary = summarize_imports(session)
    except SQLAlchemyError as error:
        console.print(f"[red]数据库操作失败：{error}[/red]")
        raise typer.Exit(code=1) from error

    render_run_stats(stats)
    render_quality_summary(summary)

    if stats.failed:
        raise typer.Exit(code=1)