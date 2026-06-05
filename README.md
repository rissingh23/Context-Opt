# Context Strategy Project

Research repo for comparing adaptive context strategies on LongBench examples.

## Setup

```bash
pip install -r requirements.txt
```

## Download LongBench Data

```bash
python -m src.data.download_longbench --limit 100
```

Options:
- `--tasks qasper hotpotqa gov_report multi_news passage_count` — tasks to download (default: all five)
- `--limit 100` — max examples per task
- `--output-dir data/longbench/processed` — output directory

## Strategies

All strategies live in `src/strategies/` and implement `prepare(example) -> StrategyResult`.

| Strategy | Description |
|---|---|
| `full_context` | Baseline — passes the full document to the answer model |
| `retrieval` | TF-IDF cosine similarity to select top-k chunks |
| `compression` | LLMLingua-2 token-level compression |
| `summarization` | Qwen local model — query-aware generative summary |
| `gemini_summarization` | Gemini Flash via Vertex AI — query-aware generative summary, no GPU needed |
| `retrieval_compression` | Retrieve top-k chunks, then compress with LLMLingua-2 |
| `retrieval_summary` | Retrieve top-k chunks, then summarize with local model |

## Running Eval

### Mock (no API key needed)

```bash
python -m src.eval_framework.run_eval_table \
  --tasks qasper \
  --strategies full_context retrieval compression \
  --limit 5 \
  --provider mock \
  --model mock_model
```

### Gemini via Vertex AI (real answers)

```bash
python -m src.eval_framework.run_eval_table \
  --tasks qasper hotpotqa gov_report multi_news passage_count \
  --strategies full_context retrieval compression gemini_summarization retrieval_compression \
  --limit 10 \
  --provider vertexai \
  --model gemini-2.5-flash \
  --vertexai-project YOUR_GCP_PROJECT \
  --vertexai-location us-central1
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--tasks` | all five | LongBench tasks to run |
| `--strategies` | all | Strategies to compare |
| `--limit` | 5 | Max examples per task |
| `--provider` | `mock` | `mock` or `vertexai` |
| `--model` | `mock_model` | Model name for the answer model |
| `--vertexai-project` | — | GCP project ID (required for `vertexai` provider) |
| `--summarization-model` | — | Override local summarization model (HuggingFace ID) |
| `--rows-output` | `outputs/processed/eval_rows.csv` | Per-example results CSV |
| `--aggregate-output` | `outputs/processed/eval_summary.csv` | Aggregated results CSV |
| `--json-output` | `outputs/processed/eval_rows.jsonl` | Checkpoint file (also used for resume) |

## Resuming an Interrupted Run

The eval runner saves results to the JSONL checkpoint file after every example. If the run is interrupted, rerun the exact same command — it will skip already-completed rows and pick up where it left off.

## Colab Notebooks

| Notebook | Description |
|---|---|
| `notebooks/colab_mock_strategy_test.ipynb` | Quick pipeline smoke test, no API keys needed |
| `notebooks/colab_hf_eval_runner.ipynb` | Full mock eval with HTML table output |
| `notebooks/colab_gemini_eval.ipynb` | Real eval using Gemini 2.5 Flash via Vertex AI |

For the Gemini notebook, authenticate with:

```python
from google.colab import auth
auth.authenticate_user()
```

## Output Format

`eval_rows.csv` — one row per (example, strategy):
- `task`, `example_id`, `strategy`, `model`
- `prediction`, `reference_answer`
- `quality_score`, `rouge_l`, `token_f1`, `exact_match`
- `original_context_tokens`, `strategy_context_tokens`, `compression_ratio`
- `strategy_latency_sec`, `model_latency_sec`, `estimated_cost`

`eval_summary.csv` — aggregated by (task, strategy, model).
