from __future__ import annotations

import json
from pathlib import Path

import polars as pl
#快速制作命令行的库
import typer
#终端美化和格式化输出库
# 替换 print，支持颜色，粗体，样式等
from rich.console import Console
# 给内容添加边框面板
from rich.panel import Panel
# 在终端打印漂亮的表格
from rich.table import Table

from enterprise_rag_cli.dataset import (
    expected_document_ids,
    file_sha256,
    load_questions,
    scan_documents,
    summarize_documents,
    summarize_questions,
)

#创建主 CLI
app = typer.Typer(help="Enterprise RAG CLI")
#创建子 CLI
dataset_app = typer.Typer(help="检查和管理 Benchmark 数据")
# 把子CLI挂载在主CLI下
app.add_typer(dataset_app, name="dataset")

console = Console()

# 使用表格打印数据
def render_frame(frame: pl.DataFrame, title: str) -> None:
    table = Table(title=title, show_lines=False)
    for column in frame.columns:
        table.add_column(column)

    for row in frame.iter_rows():
        table.add_row(
            # *的作用是元组解包
            *(
            str(value)
            for value in row
            )
        )

    console.print(table)

# 把inspect_dataset函数注册成 dataset 下的 inspect 命令
# inspect_dataset函数参数就是命令行inspect 的参数
@dataset_app.command("inspect")
def inspect_dataset(
    questions_path: Path = typer.Option(
        # 默认参数
        Path("data/enterprise-rag-bench/questions.jsonl"),
        #参数名
        "--questions",
        #路径必须存在
        exists=True,
        #允许是文件
        file_okay=True,
        #不允许是目录
        dir_okay=False,
        #要求文件可读
        readable=True,
        help="questions.jsonl 路径",
    ),
    documents_dir: Path = typer.Option(
        Path("data/enterprise-rag-bench/documents"),
        "--documents",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="文档根目录",
    ),
    report_path: Path | None = typer.Option(
        None,
        "--report",
        help="可选的 JSON 报告路径",
    ),
) -> None:
    try:
        questions = load_questions(questions_path)
        # 扫描的过程，有旋转动画
        with console.status("正在扫描文档…"):
            documents = scan_documents(documents_dir)
    except ValueError as error:
        console.print(f"[red]数据检查失败：{error}[/red]")
        raise typer.Exit(code=1) from error

    question_summary = summarize_questions(questions)
    document_summary = summarize_documents(documents)

    # 评测问题集中可用的问题占比（有些问题所需的文档没有下载，这些问题无法使用）
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
    #如果传入报告路径，那就创建报告文件
    if report_path is not None:
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


if __name__ == "__main__":
    app()