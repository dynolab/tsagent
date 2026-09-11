# code

Shared, stable code for the project. Anything here is meant to be imported by more than
one task.

Task-specific scripts live next to the task that produced them, under
[`../progress/<task>/src/`](../progress/), and graduate into this package once a second
task needs them.

## Current contents

Nothing has graduated yet. The dataset builders for the four evaluation pools still live
in [`../progress/01-anomaly-detection-agent-spec/src/`](../progress/01-anomaly-detection-agent-spec/src/):

- `sampler-r1-r2.py` — builds the real pools R1 and R2 from TSB-UAD raw files
- `generator-s1-s2.py` — generates the synthetic pools S1 and S2

## Setup

```bash
pip install -r requirements.txt
```
