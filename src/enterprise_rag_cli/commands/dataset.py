from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.panel import Panel

from enterprise_rag_cli.infrastructure.benchmark import (
    expected_document_ids,
    file_sha256,
    load_questions,
    scan_documents,
    summarize_documents,
    summarize_questions,
)

from enterprise_rag_cli.presentation import console, render_frame
from enterprise_rag_cli.config import get_settings


app = typer.Typer(no_args_is_help=True)

#inspect命令
@app.command("inspect")
def inspect_dataset(
    questions_path: Path = typer.Option(
        None,
        "--questions",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="评测问题集 questions.jsonl 的路径",
    ),
    documents_dir: Path = typer.Option(
        None,
        "--documents",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="文档库根目录",
    ),
    report_path: Path | None = typer.Option(
        None,
        "--report",
        help="可选的 JSON 报告保存路径",
    ),
) -> None:
    """检查评测问题集和当前文档库。"""

    settings = get_settings()
    questions_path = questions_path or settings.data_dir / "questions.jsonl"
    documents_dir = documents_dir or settings.data_dir / "documents"

    if not questions_path.is_file():
        console.print(f"[red]找不到评测问题集：{questions_path}[/red]")
        raise typer.Exit(code=2)

    if not documents_dir.is_dir():
        console.print(f"[red]找不到文档库目录：{documents_dir}[/red]")
        raise typer.Exit(code=2)

    try:
        questions = load_questions(questions_path)
        with console.status("正在扫描文档…"):
            documents = scan_documents(documents_dir)
    except ValueError as error:
        console.print(f"[red]数据检查失败：{error}[/red]")
        raise typer.Exit(code=1) from error

    question_summary = summarize_questions(questions)
    document_summary = summarize_documents(documents)

    expected_ids = expected_document_ids(questions)
    available_ids = {
        value
        for value in documents.get_column("document_id").drop_nulls().to_list()
    }
    covered_expected_ids = expected_ids & available_ids
    coverage = len(covered_expected_ids) / len(expected_ids) if expected_ids else 1.0

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"问题数：{questions.height:,}",
                    f"文档数：{documents.height:,}",
                    f"预期文档数：{len(expected_ids):,}",
                    f"预期文档覆盖率：{coverage:.2%}",
                    f"问题文件 SHA-256：{file_sha256(questions_path)}",
                ]
            ),
            title="EnterpriseRAG-Bench 数据概况",
        )
    )
    render_frame(question_summary, "问题类型")
    render_frame(document_summary, "文档来源")

    if coverage < 1:
        console.print(
            "[yellow]当前文档库没有覆盖全部预期文档。"
            "评测前需要筛选可用问题。[/yellow]"
        )

    if report_path is None:
        return

    report = {
        "questions_path": str(questions_path),
        "documents_dir": str(documents_dir),
        "questions_sha256": file_sha256(questions_path),
        "question_count": questions.height,
        "document_count": documents.height,
        "expected_document_count": len(expected_ids),
        "covered_expected_document_count": len(covered_expected_ids),
        "expected_document_coverage": coverage,
        "question_types": question_summary.to_dicts(),
        "document_sources": document_summary.to_dicts(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]已写入报告：{report_path}[/green]")