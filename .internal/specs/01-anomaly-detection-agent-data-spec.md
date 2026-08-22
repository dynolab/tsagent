## 01-anomaly-detection-agent-data-spec

### 1. Executive summary

#### 1.1 Spec description

Collect four datasets (2 real industrial + 2 synthetic) for time series anomaly detection, each containing approximately 1,000 univariate series with series-level binary labels. This data collection supports RQ1 of the tsagent constitution spec: comparing a tool-augmented LLM agent against a raw-retrieval-only agent in anomaly detection quality. The datasets provide the empirical basis for evaluation — they are not analyzed for insights, only prepared with stable labels and formats.

#### 1.2 Spec motivation

No existing benchmark provides the combination needed for this project: large pools of industrial univariate time series with series-level anomaly labels, paired with synthetic datasets where anomaly types and injection rates are controlled. Public datasets either lack series-level labeling or are too small for pool-level evaluation. Synthetic generators are necessary to control for anomaly type, rate, and severity, enabling systematic testing of tool and agent behavior.

### 2. Data specification

#### 2.1 Schema

Each dataset is stored as a parquet file with the following columns:
- `series_id` (string): unique stable identifier (format: `{group}__{source}__{series_idx}_full`)
- `time_index` (int): ordered, monotonic integer index (no timestamps — relative indexing)
- `value` (float64): observed value
- `label` (int8): point-level binary anomaly flag (1 = anomalous point, 0 = normal)

Per-dataset metadata is stored as a separate parquet file (`{group}_metadata.parquet`) with one row per series:
- `series_id`, `length`, `num_point_anomalies`, `anomaly_fraction`
- `y_i` (int): series-level binary label (1 if any point anomaly present, 0 otherwise)
- `is_split` (bool): whether the series was split during label derivation
- `original_length`, `base_type` (synthetic only), `anomaly_type` (synthetic only), `severity`, `target_fraction`

#### 2.2 Volume and scale

- **R1 (Combined IT)**: YAHOO, SMD, IOPS, Exathlon, WSD, NEK from TSB-UAD-Public-v2 — web traffic, server KPIs, Spark logs, network flows
- **R2 (Combined Biomedical)**: MITDB, SVDB, LTDB from TSB-UAD-Public-v2 — ECG records (arrhythmias, long-term monitoring)
- **S1 (Stationary)**: 500 series × 1000 points each
- **S2 (Trend-Seasonal)**: 500 series × 1000 points each
- Synthetic anomaly rate: 50% of series are anomalous, with 2.5–5% of points flagged per anomalous series

#### 2.3 Sampling strategy

- **Real datasets**: Selected from TSB-UAD-Public-v2 (https://www.thedatum.org/datasets). R1 combines IT-Ops datasets (YAHOO, SMD, IOPS, Exathlon, WSD, NEK) for diverse industrial monitoring patterns. R2 combines biomedical ECG datasets (MITDB, SVDB, LTDB). Both have point-level anomaly labels covering various anomaly types: point spikes (amplitude), collective/group, trend, frequency/rhythm.
- **Synthetic datasets**: Generated programmatically via `generator-s1-s2.py`. S1 uses stationary base processes (white noise, AR(1), AR(2)) with anomaly types: point, group, level_shift, variance. S2 uses trend-seasonal base processes (linear trend, seasonal sine, trend+seasonal) with anomaly types: trend, seasonality, group, level_shift. Severity range: 2.6–4 standard deviations. Random seed fixed at 42 for reproducibility.

### 3. Collection / generation protocol

- **Real datasets**: Downloaded from TSB-UAD-Public-v2 (https://www.thedatum.org/datasets/TSB-UAD-Public-v2.zip). Each source dataset is a CSV file with `Data` and `Label` columns (point-level annotations). Combined into the unified parquet schema described in 2.1.
- **Synthetic datasets**: Generated via `generator-s1-s2.py` with fixed random seed (42). Each generator calls `generate_synthetic_pool()` with the parameters specified in 2.3. Outputs two parquet files per dataset: `{group}.parquet` (series data) and `{group}_metadata.parquet` (per-series metadata).

### 4. Quality criteria

- **Completeness**: No missing values in `value` column — all series must have continuous `time_index` from 0 to `length-1`.
- **Label accuracy**: Point-level labels from TSB-UAD sources are used as-is (trusted ground truth). For synthetic data, labels are verified by construction: injected anomaly regions match the `label` column exactly.
- **Duplicate removal**: Series with identical `series_id` are rejected. Synthetic generator ensures unique IDs by construction (`{group}__SYNTH__{base_type}__{series_idx}_full`).
- **Outlier handling**: No outlier removal — anomalies are the signal of interest. The label field distinguishes anomalous from normal points.
- **Reproducibility**: Synthetic generation uses fixed random seed (42) for deterministic output. Rerunning `generator-s1-s2.py` produces identical parquet files.

### 5. Annotation / labeling plan (if applicable)

#### 5.1 Labeling scheme

Point-level binary labels (`label`: 0 = normal, 1 = anomalous). Series-level binary labels (`y_i`: 0 = normal series, 1 = anomalous series) derived from point labels: `y_i = 1` if any point in the series has `label = 1`, else `0`.

#### 5.2 Annotation process

- **Real datasets**: Labels provided by TSB-UAD-Public-v2. No manual annotation. Point-level annotations are taken from the `Label` column of source CSVs.
- **Synthetic datasets**: Labels generated programmatically during anomaly injection. The `inject_anomaly()` function marks affected time indices as `label = 1`. Series-level `y_i` computed as the logical OR over all point labels.

#### 5.3 Quality control

- Synthetic labels are verified by construction — the injection function directly writes the label array.
- Real dataset labels are trusted as ground truth from the benchmark source.
- Series-level `y_i` is derived mechanically from point labels (no human judgment involved).

### 6. Implementation plan

#### 6.1 Implementation repos

[List the repos expected to be involved in implementation]

#### 6.2 Deliverables

[Dataset files, schema documentation, collection scripts, quality report, annotation guidelines]

#### 6.3 Todo list

[High-level steps to complete this spec]
