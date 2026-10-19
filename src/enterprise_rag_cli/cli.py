from __future__ import annotations

import typer

from enterprise_rag_cli.commands.dataset import app as dataset_app
from enterprise_rag_cli.commands.version import show_version
from enterprise_rag_cli.commands.ingest import ingest


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


if __name__ == "__main__":
    app()
