## Milestone: M1 — Baseline development + dataset preparation

**T = 2026-08-01**

Constitution roadmap ref: M1 expected result — "Raw-retrieval-only baseline agent (single tool providing time series access), dataset pool builder, label derivation for selected datasets" (T+2, strong scaling efficiency 0.65)

### Task table

| ID | Task name | Status | Duration (weeks) | Spec ref | Notes |
|----|-----------|--------|------------------|----------|-------|
| 01 | Build real dataset R1+R2 (download, unify schema) | Done | 2 | 01-anomaly-detection-agent-data-spec.md | TSB-UAD-Public-v2 -> unified parquet |
| 02 | Build synthetic generators S1+S2 | Done | 2 | 01-anomaly-detection-agent-data-spec.md | stationary + trend-seasonal, seed=42 |
| 03 | Label derivation + quality checks across R1/R2/S1/S2 | Done | 1 | 01-anomaly-detection-agent-data-spec.md | y_i derivation, completeness/dup checks |
| 04 | Build raw-retrieval-only baseline agent (tools + pydantic-ai harness + single-episode run producing anomaly_verdicts.jsonl and decision_trace.jsonl) | Done | 1 | 02-raw-retrieval-baseline-agent-infrastructure-spec.md | two tools: get_series, get_pool_meta_data; no anomaly-detection tools; code/src/baseline_agent, write-up in progress/02-raw-retrieval-baseline-agent-infrastructure |

- **Status**: `Todo`, `Doing`, `Done`
- **Spec ref**: filename in `.internal/specs/`
- **Duration**: planned (for status `Todo` and `Doing`) or factual (for status `Done`) duration in weeks; can be 1, 2 or 3 (no more)

---

### Backlog tasks

| Task name | Notes |
|-----------|-------|
| | |
| | |
