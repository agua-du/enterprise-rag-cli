from __future__ import annotations

import typer
from rich.table import Table

from enterprise_rag_cli.config import get_settings
from enterprise_rag_cli.infrastructure.database import create_session_factory
from enterprise_rag_cli.presentation import console
from enterprise_rag_cli.application.batch_indexing import (
    apply_full_index,
    prepare_full_index,
    refresh_batch_status,
    submit_batch_shards,
)


app = typer.Typer(no_args_is_help=True)


@app.command("batch-prepare")
def batch_prepare() -> None:
    """分页读取完整文档库并生成 Batch 分片。"""
    settings = get_settings()
    session_factory = create_session_factory(settings)

    with console.status("正在切分文档并生成 Batch 分片…"):
        with session_factory() as session:
            run = prepare_full_index(session, settings)

    console.print(f"Run ID：{run.run_id}")
    console.print(f"文档：{run.documents}")
    console.print(f"Chunk：{run.chunks}")
    console.print(f"分片：{run.shards}")
    console.print(f"本地任务目录：{run.run_dir}")


@app.command("batch-submit")
def batch_submit(run_id: str) -> None:
    """提交尚未完成的 Batch 分片。"""
    submitted = submit_batch_shards(get_settings(), run_id)
    console.print(f"本次提交分片：{submitted}")


@app.command("batch-status")
def batch_status(run_id: str) -> None:
    """汇总所有 Batch 分片的状态。"""
    counts = refresh_batch_status(get_settings(), run_id)
    table = Table(title=f"全量索引任务 {run_id}")
    table.add_column("状态")
    table.add_column("分片", justify="right")
    for status, count in sorted(counts.items()):
        table.add_row(status, str(count))
    console.print(table)


@app.command("batch-apply")
def batch_apply(run_id: str) -> None:
    """应用已完成分片，并在全部完成后切换 Alias。"""
    with console.status("正在应用已经完成的向量分片…"):
        result = apply_full_index(get_settings(), run_id)

    console.print(
        f"已应用分片：{result.applied_shards}/{result.total_shards}"
    )
    console.print(f"Collection：{result.collection_name}")
    if result.already_active:
        console.print("这个索引版本已经启用。")
    elif result.activated:
        console.print("全部分片校验通过，完整索引已经启用。")
    else:
        console.print("仍有分片未完成，当前 Alias 保持不变。")
