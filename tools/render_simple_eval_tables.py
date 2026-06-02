"""Render compact presentation tables from eval_summary.csv."""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


TASK_ORDER = ["qasper", "hotpotqa", "gov_report", "multi_news", "passage_count"]
TASK_LABELS = {
    "qasper": "Qasper",
    "hotpotqa": "HotpotQA",
    "gov_report": "GovReport",
    "multi_news": "MultiNews",
    "passage_count": "PassageCount",
}
STRATEGY_LABELS = {
    "full_context": "Full Context",
    "retrieval": "Retrieval",
    "compression": "Compression",
    "summarization": "Summarization",
    "retrieval_compression": "Retrieval + Compression",
    "retrieval_summary": "Retrieval + Summary",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except ValueError:
        return None


def fmt_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def fmt_percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def fmt_tokens(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def fmt_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}s"


def fmt_cost(value: float | None) -> str:
    return "-" if value is None else f"${value:.4f}"


def avg(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def strategy_name(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def task_name(name: str) -> str:
    return TASK_LABELS.get(name, name)


def build_quality_matrix(rows: list[dict[str, str]], score_column: str) -> list[list[str]]:
    by_strategy_task: dict[str, dict[str, float | None]] = defaultdict(dict)
    for row in rows:
        by_strategy_task[row["strategy"]][row["task"]] = to_float(row.get(score_column, ""))

    table = [["Strategy", *[task_name(task) for task in TASK_ORDER], "Avg"]]
    for strategy in sorted(by_strategy_task):
        task_scores = [by_strategy_task[strategy].get(task) for task in TASK_ORDER]
        table.append([strategy_name(strategy), *[fmt_score(score) for score in task_scores], fmt_score(avg(task_scores))])
    return table


def build_efficiency_table(rows: list[dict[str, str]]) -> list[list[str]]:
    by_strategy: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_strategy[row["strategy"]].append(row)

    table = [["Strategy", "Avg Input Tok", "Avg Ctx %", "Avg Latency", "Avg Cost", "Avg Utility"]]
    for strategy in sorted(by_strategy):
        group = by_strategy[strategy]
        input_tokens = avg([to_float(row.get("avg_input_tokens", "")) for row in group])
        compression = avg([to_float(row.get("avg_compression_ratio", "")) for row in group])
        latency = avg([to_float(row.get("avg_total_latency_sec", "")) for row in group])
        cost = avg([to_float(row.get("avg_estimated_cost", "")) for row in group])
        utility = avg([to_float(row.get("avg_utility_score", "")) for row in group])
        table.append(
            [
                strategy_name(strategy),
                fmt_tokens(input_tokens),
                fmt_percent(compression),
                fmt_seconds(latency),
                fmt_cost(cost),
                fmt_score(utility),
            ]
        )
    return table


def render_markdown_table(table: list[list[str]]) -> str:
    header = table[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in table[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_html_table(table: list[list[str]]) -> str:
    header = "".join(f"<th>{html.escape(cell)}</th>" for cell in table[0])
    rows = []
    for row in table[1:]:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_html(title: str, quality_table: list[list[str]], efficiency_table: list[list[str]], score_column: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 32px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f8fa;
      color: #17202a;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 18px;
    }}
    p {{
      margin: 0 0 18px;
      color: #5d6977;
    }}
    table {{
      border-collapse: collapse;
      min-width: 760px;
      background: #fff;
      box-shadow: 0 1px 4px rgba(17, 24, 39, 0.08);
    }}
    th, td {{
      border: 1px solid #d9e0e7;
      padding: 11px 13px;
      font-size: 14px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      font-weight: 650;
    }}
    th {{
      background: #edf2f7;
    }}
    tr:nth-child(even) td {{
      background: #fafbfc;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Quality table uses <code>{html.escape(score_column)}</code>. Mock-model scores are placeholders until real answer/judge models are connected.</p>
  <h2>Quality By Task</h2>
  {render_html_table(quality_table)}
  <h2>Efficiency Summary</h2>
  {render_html_table(efficiency_table)}
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/processed/eval_summary.csv"))
    parser.add_argument("--score-column", default="avg_quality_score")
    parser.add_argument("--markdown-output", type=Path, default=Path("outputs/figures/simple_eval_tables.md"))
    parser.add_argument("--html-output", type=Path, default=Path("outputs/figures/simple_eval_tables.html"))
    parser.add_argument("--title", default="Context Strategy Comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    quality_table = build_quality_matrix(rows, args.score_column)
    efficiency_table = build_efficiency_table(rows)

    markdown = "\n\n".join(
        [
            f"# {args.title}",
            f"Quality score column: `{args.score_column}`",
            "## Quality By Task",
            render_markdown_table(quality_table),
            "## Efficiency Summary",
            render_markdown_table(efficiency_table),
        ]
    )

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown + "\n", encoding="utf-8")
    args.html_output.write_text(render_html(args.title, quality_table, efficiency_table, args.score_column), encoding="utf-8")

    print(f"Wrote simple Markdown tables to {args.markdown_output}")
    print(f"Wrote simple HTML tables to {args.html_output}")


if __name__ == "__main__":
    main()
