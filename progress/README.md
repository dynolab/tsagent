# Progress

Working log of the project, one directory per task. Each directory is self-contained: it
holds the task's own spec, the code that produced its artefacts, and a README stating what
is done and how to rebuild it.

Numbering is chronological, so the directory list doubles as the project timeline.

| Task | Status | Summary |
|------|--------|---------|
| [01-anomaly-detection-agent-spec](01-anomaly-detection-agent-spec/) | data complete, agent spec open | Task contract for the anomaly-detection MVP, plus the four evaluation pools it runs on: R1/R2 sampled from TSB-UAD and S1/S2 generated from scratch (13,942 series total). |

## Conventions

- **Generated data is never committed.** Pools, raw inputs and intermediate parquet files
  are git-ignored; every artefact is reproducible from the scripts in the task directory.
- **Scripts are configuration-driven.** Each has a `CONFIGURATION` block at the top holding
  every knob, and no command-line arguments — edit the block, run the file.
- **Output is validated before it is written.** A broken invariant aborts the run rather
  than producing a file that later stages would trust.
- Stable, reusable code graduates from a task directory into [`../code/`](../code/).
