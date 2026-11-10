from __future__ import annotations

import typer

from enterprise_rag_cli.commands.dataset import app as dataset_app
from enterprise_rag_cli.commands.version import show_version
from enterprise_rag_cli.commands.ingest import ingest
from enterprise_rag_cli.commands.index import app as index_app
from enterprise_rag_cli.commands.chat import chat
from enterprise_rag_cli.commands.benchmark import app as benchmark_app
from enterprise_rag_cli.commands.chunks import app as chunks_app


app = typer.Typer(
    name="rag",
    help="Enterprise RAG CLI",
    no_args_is_help=True,
)

app.add_typer(
    dataset_app,
    name="dataset",
    help="检查和管理 EnterpriseRAG-Bench 数据",
)
app.command("version")(show_version)
app.command("ingest")(ingest)

app.add_typer(
    index_app,
    name="index",
    help="建立和管理检索索引"
)
app.command("chat")(chat)
app.add_typer(benchmark_app, name="benchmark", help="评测与结果分析")
app.add_typer(chunks_app, name="chunks", help="文档切分与预览")

if __name__ == "__main__":
    app()
