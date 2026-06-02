"""Render eval CSV outputs as readable Markdown and HTML tables."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "task",
    "strategy",
    "model",
    "num_examples",
    "error_rate",
    "avg_quality_score",
    "avg_token_f1",
    "avg_rouge_l",
    "avg_numeric_exact_match",
    "avg_input_tokens",
    "avg_compression_ratio",
    "avg_total_latency_sec",
    "avg_estimated_cost",
    "avg_utility_score",
]


DISPLAY_NAMES = {
    "task": "Task",
    "strategy": "Strategy",
    "model": "Model",
    "num_examples": "N",
    "error_rate": "Err %",
    "avg_quality_score": "Quality",
    "avg_token_f1": "F1",
    "avg_rouge_l": "ROUGE-L",
    "avg_numeric_exact_match": "Num EM",
    "avg_input_tokens": "Input Tok",
    "avg_compression_ratio": "Ctx %",
    "avg_total_latency_sec": "Latency",
    "avg_estimated_cost": "Cost",
    "avg_utility_score": "Utility",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except ValueError:
        return None


def format_cell(column: str, value: str) -> str:
    number = to_float(value)
    if number is None:
        return value
    if column in {"error_rate", "avg_compression_ratio"}:
        return f"{number * 100:.1f}%"
    if column in {"avg_input_tokens", "num_examples"}:
        return f"{number:,.0f}"
    if column == "avg_estimated_cost":
        return f"${number:.4f}"
    if column == "avg_total_latency_sec":
        return f"{number:.2f}s"
    return f"{number:.3f}"


def select_columns(rows: list[dict[str, str]], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    available = set(rows[0]) if rows else set()
    return [column for column in DEFAULT_COLUMNS if column in available]


def render_markdown(rows: list[dict[str, str]], columns: list[str]) -> str:
    headers = [DISPLAY_NAMES.get(column, column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        cells = [format_cell(column, row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def render_html(rows: list[dict[str, str]], columns: list[str], title: str) -> str:
    headers = "".join(f"<th>{html.escape(DISPLAY_NAMES.get(column, column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(format_cell(column, row.get(column, '')))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
      background: #f7f8fa;
    }}
    h1 {{
      font-size: 24px;
      margin-bottom: 16px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
    }}
    th, td {{
      border: 1px solid #dde3ea;
      padding: 10px 12px;
      text-align: left;
      font-size: 14px;
      white-space: nowrap;
    }}
    th {{
      background: #edf2f7;
      font-weight: 700;
    }}
    tr:nth-child(even) td {{
      background: #fafbfc;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>
      {"".join(body_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/processed/eval_summary.csv"))
    parser.add_argument("--markdown-output", type=Path, default=Path("outputs/figures/eval_summary_table.md"))
    parser.add_argument("--html-output", type=Path, default=Path("outputs/figures/eval_summary_table.html"))
    parser.add_argument("--title", default="Context Strategy Evaluation Summary")
    parser.add_argument("--columns", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    columns = select_columns(rows, args.columns)

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)

    args.markdown_output.write_text(render_markdown(rows, columns), encoding="utf-8")
    args.html_output.write_text(render_html(rows, columns, args.title), encoding="utf-8")

    print(f"Wrote Markdown table to {args.markdown_output}")
    print(f"Wrote HTML table to {args.html_output}")


if __name__ == "__main__":
    main()
