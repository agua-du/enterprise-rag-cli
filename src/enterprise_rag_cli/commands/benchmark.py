from pathlib import Path

import typer
from rich.table import Table
import json

from enterprise_rag_cli.application.benchmark import prepare_benchmark
from enterprise_rag_cli.config import get_settings
from enterprise_rag_cli.presentation import console
from enterprise_rag_cli.application.benchmark import run_benchmark
from enterprise_rag_cli.application.benchmark_report import (
    load_evaluated_run,
    save_review_cases,
    summarize_results,
    compare_benchmarks
)


app = typer.Typer(help="运行评测并分析结果", no_args_is_help=True)


@app.command("prepare")
def prepare(
    output: Path = typer.Option(
        Path("benchmark/suites/smoke-5k"),
        "--output",
        help="固定评测集的保存目录",
    ),
) -> None:
    """筛选当前索引能够支持的评测问题。"""
    suite = prepare_benchmark(get_settings(), output)
    console.print(f"索引中的文档：{suite['document_count']}")
    console.print(f"保留的问题：{suite['question_count']}")
    console.print(f"评测集已保存：{output}")

@app.command("run")
def run(
    suite: Path = typer.Option(
        Path("benchmark/suites/smoke-5k"), "--suite",
        exists=True, file_okay=False,
    ),
    output: Path = typer.Option("reports/benchmark/baseline-k5", "--output", help="本次运行的保存目录"),
    top_k: int = typer.Option(5, "--top-k", min=1, max=20),
) -> None:
    """批量回答固定评测集中的问题。"""
    with console.status("正在运行评测问题并保存证据…"):
        manifest = run_benchmark(get_settings(), suite, output, top_k)
    console.print(f"问题数：{manifest['question_count']}")
    console.print(f"调用失败：{manifest['errors']}")
    console.print(f"结果目录：{output}")
    if manifest["errors"]:
        raise typer.Exit(code=1)

@app.command("report")
def report(run_dir: Path = typer.Argument(..., exists=True, file_okay=False)):
    """汇总评分，并导出需要人工检查的样本。"""
    _, _, rows = load_evaluated_run(run_dir)
    summary = summarize_results(rows)
    table = Table(title="RAG 评测报告")
    for name in ("类型", "题数", "正确率%", "完整性%", "文档召回%", "命中率%", "MRR", "平均秒"):
        table.add_column(name)
    for row in summary.iter_rows(named=True):
        table.add_row(
            row["question_type"], str(row["questions"]),
            f"{row['correctness_pct']:.2f}",
            f"{row['completeness_pct']:.2f}",
            f"{row['document_recall_pct']:.2f}",
            f"{row['hit_rate_pct']:.2f}",
            f"{row['mrr']:.3f}",
            f"{row['latency_seconds']:.3f}",
        )
    console.print(table)
    summary.write_csv(run_dir / "summary.csv")
    count = save_review_cases(rows, run_dir / "review.jsonl")
    console.print(f"待检查样本：{count}，已保存到 {run_dir / 'review.jsonl'}")

@app.command("compare")
def compare(
    baseline: Path = typer.Argument(..., exists=True, file_okay=False),
    candidate: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """比较同一批问题的两次完整评测。"""
    rows = compare_benchmarks(baseline, candidate)
    table = Table(title="逐题变化（候选版本减去基线）")
    for name in ("问题", "正确性", "完整性百分点", "召回百分点", "耗时秒"):
        table.add_column(name)
    for row in rows:
        table.add_row(
            row["question_id"], f"{row['correctness_delta']:+d}",
            f"{row['completeness_delta']:+.2f}",
            f"{row['recall_delta']:+.2f}",
            f"{row['latency_delta']:+.3f}",
        )
    console.print(table)
    improved = sum(row["correctness_delta"] > 0 for row in rows)
    regressed = sum(row["correctness_delta"] < 0 for row in rows)
    console.print(f"由错变对：{improved}；由对变错：{regressed}")
    path = candidate / "comparison.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    console.print(f"比较结果：{path}")
