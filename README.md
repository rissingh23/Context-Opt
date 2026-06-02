# Context Strategy Project

Research repo for comparing adaptive context strategies on LongBench examples.

## Starter Dataset

The initial LongBench tasks are:

- `qasper`
- `hotpotqa`
- `gov_report`
- `multi_news`
- `passage_count`

Download and normalize a small local copy:

```bash
python -m src.data.download_longbench --limit 100
```

Load examples from Python:

```python
from src.data.load_examples import load_examples

examples = load_examples(tasks=["qasper"], limit=5)
```

Each normalized JSONL row includes:

- `task`
- `example_id`
- `query`
- `context`
- `reference_answer`
- `answers`
- `task_type`
- `context_length_chars`
- `metadata`

## Strategy Modules

Strategy files live in `src/strategies/` and expose a common interface through
`ContextStrategy.prepare(example)`. The current files are placeholders ready for
implementation:

- `full_context.py`
- `retrieval.py`
- `summarization.py`
- `compression.py`
- `hybrid.py`
