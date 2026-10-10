# code

Shared, stable code for the project. Anything here is meant to be imported by more than
one task.

Task-specific scripts live next to the task that produced them, under
[`../progress/<task>/src/`](../progress/), and graduate into this package once a second
task needs them.

## Current contents

- [`src/baseline_agent/`](src/baseline_agent/) - the raw-retrieval-only baseline agent of
  task 02: pool storage, the two retrieval tools, the pydantic-ai agent, the episode
  runner and the writers of `anomaly_verdicts.jsonl` / `decision_trace.jsonl`. It is the
  reference point every later agent (M2/M3) is compared against, so it lives here from
  the start. Module map and design notes in
  [`../progress/02-raw-retrieval-baseline-agent-infrastructure/`](../progress/02-raw-retrieval-baseline-agent-infrastructure/).
- [`src/run_baseline_agent.py`](src/run_baseline_agent.py) - configuration-driven entry
  point that runs one episode (or several replays) over a pool.
- [`tests/`](tests/) - unit tests for the package; no LLM, no GPU, runs in CI.

The dataset builders for the four evaluation pools still live in
[`../progress/01-anomaly-detection-agent-spec/src/`](../progress/01-anomaly-detection-agent-spec/src/):

- `sampler-r1-r2.py` - builds the real pools R1 and R2 from TSB-UAD raw files
- `generator-s1-s2.py` - generates the synthetic pools S1 and S2

## Setup

```bash
pip install -r requirements.txt
```

## Tests

```bash
cd code
python -m pytest
```
