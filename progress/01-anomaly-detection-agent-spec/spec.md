# Anomaly Detection Agent Spec

## One-Paragraph Spec
The MVP functions as an offline agent that orchestrates classical time-series analysis tools through an LLM interface. It processes input pools containing thousands of real and synthetic univariate series stored in parquet files with explicit labels. The agent avoids direct raw data interpretation, instead planning budgeted calls to deterministic algorithms like anomaly detection and decomposition. A hard constraint ensures that all generated narratives are strictly grounded in the structured outputs provided by these tools. Consequently, the system produces a ranked list of potential anomalies accompanied by human-readable explanations. Each explanation details the nature, timing, and severity of detected issues while maintaining traceability to the underlying evidence. This approach allows for efficient triage of large datasets, directing human attention only to actionable events. The result is a cost-effective workflow that generates faithful, uncertainty-aware insights across various industrial and financial domains.

- **Input**: Pool of 5,800 real univariate series (R1 + R2, 2,900 each; uncomposed full variants with 4,082 and 12,581 samples ship alongside) and 4,000 synthetic univariate series (S1 + S2, 2,000 each) stored as parquet files with columns `series_id`, `time_index`, `value`, `label` (point-wise anomaly flag)
- **Output**: ranked list of anomalous series + human-readable explanation why each time series is deemed anomalous
- **Core idea**: LLM agent plans budgeted deep-dives using deterministic tools; narrative must be evidence-linked
- **Hard constraint**: agent must not make claims not supported by tool outputs

## Scope

### In scope (must-haves)
- Anomaly detection only
- Retrieve abnormal series from a pool of ~1,000 univariate series
- Full-series analysis (not streaming)
- Two-stage process:
  1) cheap scan across all series
  2) deep detect for a limited subset (budgeted)
- LLM tool-calling agent required: selects which series to deep-dive and which preset to use and then provides explanation for each anomalous time series
- Evaluation across 2 real industrial datasets + 2 synthetic datasets

### Out of scope (explicitly excluded)
- Streaming/online interface
- Computational issues
- Multivariate time series

## Datasets Commitment

### Real datasets (2)
| ID | Dataset name | Source / link | Unit of analysis | Sampling unit | Length (in sampling units) | Expected #series available | Notes / constraints |
|----|--------------|---------------|------------------|---------------|----------------------------|----------------------------|---------------------|
| R1 | Combined IT (YAHOO, SMD, IOPS, Exathlon, WSD, NEK) | https://www.thedatum.org/datasets/TSB-UAD-Public-v2.zip | univariate series | point index / 1-minute / irregular (processed in TSB) | 1,009 - 12,960 (mean 2,512) | 2,900 composed (4,082 full) | The largest industrial IT-Ops pool. Web traffic, servers, KPIs, Spark logs, network flows |
| R2 | Combined Biomedical (MITDB, SVDB, LTDB) | https://www.thedatum.org/datasets/TSB-UAD-Public-v2.zip | univariate series | point index / irregular (processed in TSB) | 1,010 - 1,696 (mean 1,559) | 2,900 composed (12,581 full) | Purely medical pool - ECG records (arrhythmias, long-term monitoring) |

- **Spare datasets**: SWaT, GECCO, CreditCard

### Synthetic generators (2)
| ID | Generator name | Base process | Anomaly families injected | Controls (rate/severity/duration) |
|----|----------------|-------------|----------------------------|----------------------------------|
| S1 | Stationary | white_noise, ar1, ar2 | point, group, level_shift, variance, flatline | 2,000 series, length 1,000-4,000, rate=0.25, severity dial 2.6-3.6·σ_vis (point 2.6-3.1, group 2.8-3.6), anomalous fraction 1-20% log-uniform (point: 0.4-2.5%) |
| S2 | Trend-Seasonal | linear_trend, seasonal_sine, trend_seasonal | trend, seasonality, group, level_shift, flatline, point | 2,000 series, length 1,000-4,000, rate=0.25, severity dial 2.6-3.6·σ_vis (point 2.6-3.1, group 2.8-3.6), anomalous fraction 1-20% log-uniform (point: 0.4-2.5%) |

- **Anomalous-point fraction** is drawn **log-uniformly** from [0.01, 0.20]: real pools span two orders of magnitude (R1 median ≈ 0.010, R2 median ≈ 0.125), so equal coverage per order of magnitude gives quality-vs-density curves support across the whole axis instead of a narrow band. Both the drawn target and the realised fraction are recorded per series.
- **Base-process realism** - the "normal" regime is deliberately harder than textbook processes, so detectors cannot pass by assuming iid Gaussian noise:
  - ~25% of series draw **Student-t innovations** (df 4-10, variance-normalised, redrawn beyond 3.5σ): heavy tails make the normal process itself produce frequent 2.5-3.5σ shoulder excursions - hard negatives - while the truncation guarantees a *clean* series never manufactures a lone 8-29σ spike bigger than the labelled anomalies;
  - noise under trend/seasonal bases is **AR(1)-coloured** (φ up to 0.7), matching autocorrelated residuals of real monitoring data;
  - ~35% of seasonal bases carry a **nested second seasonality** (a 4-8× longer, weaker cycle) and ~40% a **slow amplitude modulation** (±10-30% over the series), so "the envelope changed" is not per se anomalous;
  - every series sits at a random **baseline offset** in [-50, 50], catching zero-mean assumptions;
  - ~25% of dirty non-point series carry **two disjoint events** of the same family (≥25 clean samples apart), so "found one anomaly" is not "found them all".
  Every drawn parameter lands in `base_params`.

- **Severity unit**: σ is `sigma_vis`, the series' **local visible band** - the robust spread (MAD·1.4826) of the clean series around a slow rolling-median baseline whose window exceeds the seasonal period. The trend is excluded from the unit (a global spread would be dominated by it), the seasonal swing is included (an anomaly on a wave must compete with the wave), and slow AR wander is excluded (a spike must beat the *local* crowd). A difference-based scale is **not** used: it is blind to the seasonal swing and runs ~3× the innovation scale on oscillatory AR bases, mis-sizing anomalies in both directions. Two more visible-geometry statistics are measured per series and drive the adaptive sizing below: the **reach** (q99.5 of the |residual| - how far the series itself strays from its baseline) and the **tail step** (q99.9 - q99.5 - how fast its tail keeps going; wide under Student-t noise, near zero for a bounded wave).
- **Base-process parameters** are drawn per series (AR coefficients from the stationarity triangle, seasonal period 12-200 with a second harmonic, trend parameterised by total rise) and recorded in `base_params` in the metadata.
- **Anomaly family definitions** - the six mechanisms are deliberately distinct so that no two collapse into the same signal:

| Family | Mechanism | Distinguishing property |
|--------|-----------|-------------------------|
| `point` | isolated outliers anchored at `baseline ± d`, `d = reach + margin(severity, tail)` - always beyond the farthest the series itself strays | minimum spacing of 5 points, a 50-point edge keep-out, per-spike upward jitter, own sparse density range (0.4-2.5%) |
| `group` | an alien deterministic waveform (square / triangle / sawtooth) is laid over the segment | the *shape* stops belonging to the generating process |
| `level_shift` | sustained mean step of severity·σ | sharp 2-sample edges, mean displaced, waveform preserved |
| `variance` | local dispersion scaled by (1 + severity/2): the existing fluctuation is multiplied, not resampled | mean, waveform, autocorrelation and tail law preserved - exactly one property (dispersion) changes |
| `trend` | transient linear drift peaking at severity·σ | returns to zero by the segment end |
| `seasonality` | the seasonal period is warped in place, and the resulting discrepancy is calibrated to span severity·σ_vis peak-to-peak | genuine frequency change, not an extra sine on top - and calibrated to be *visible*, not just spectrally present |
| `flatline` | the segment collapses onto a constant held value (stuck sensor) | the only family that *removes* variation instead of adding any |

- **Sizing and detectability certification** - the sizing is adaptive to what each series itself does, the certification statistical:
  - **Sizing is proportional and adaptive in both directions.** Segment families are sized as `severity × sigma_vis` (`level_shift`/`trend` displace by it, `group`/`seasonality` span it peak-to-peak): a quiet series gets small anomalies, a wide or wavy one gets large ones. Each family is then floored against the masking its own series produces and capped when the drawn size would dwarf what the series itself does:
    - `group` / injected oscillation: amplitude within [½·reach, ½·reach + ¾·σ_vis] - below it the shapelet drowns in a wide band, above it it is a caricature. The alien waveform draws its period in samples (12-40, ≥ 2.5 cycles), so it is never absorbed by a rolling baseline as level steps and never blends into the noise;
    - `level_shift` / `trend`: capped just above the reach, floored against the baseline's own natural wander at the event's horizon (at most 3× the drawn size; legibility wins over the cap);
    - `variance`: the inflation must exceed the series' natural q99.5 rolling-scale swing (capped at 2.5×);
    - `seasonality` (warped): the discrepancy is calibrated - up *or* down - to span `severity × sigma_vis` peak-to-peak, so a warp the noise buries is amplified and an over-loud one is tempered;
    - `point`: every spike lands at `baseline ± d`, `d = reach + margin(severity, tail step)` - anchored to the baseline, because a spike merely *added* to the current sample can land at the mean when the sample sat at the opposite extreme (a sign that would do so is flipped to the far side).

    Per-family severity ranges go in `SEVERITY_RANGE_OVERRIDES`, per-family density ranges in `ANOMALY_FRACTION_OVERRIDES`.
  - **Certification is statistical and only raises**: every injected event must reach a whitened matched-filter z of `SEGMENT_SNR_FLOOR = 5` against the series' known AR polynomial and innovation scale - for `point`, aggregated over the whole spike pattern. A too-weak magnitude is raised to the line; nothing below the drawn severity ever survives.
  - **duration floors** give sample-size-hungry families the points their test statistic needs (`variance` 64, `group` 64, `trend` 48, `seasonality` 48 and ≥ 2 full periods, `flatline` 24, `level_shift` 24); an event that cannot be placed leaves the series clean and honestly marked as such;
  - **`point` is sparse by definition**: its density has its own range (0.4-2.5% instead of the pool-wide 1-20%) - at the pool ceiling isolated spikes stop being isolated and form their own population;
  - on periodic bases, **`level_shift` and `trend` span ≥ 2 full periods**: anything shorter rides a single wave flank and reads as the wave's own motion or as the amplitude modulation the clean bases are allowed to have;
  - a **warped `seasonality` span accumulates ≥ 1 full cycle of discrepancy** (the duration grows, or the warp strengthens when the series is too short) - a fraction of a cycle is a phase wobble, not a frequency change;
  - **`flatline`** is certified structurally: ≥ ~18 exactly-constant samples on a series whose local noise is alive - a zero-diff run no stochastic base produces; its recorded z is a conservative lower bound that does not credit this run-length signature.

  The drawn `severity` stays the difficulty axis; the magnitude actually applied after adaptation (`severity_effective` - genuinely per-series: larger than drawn on wide or heavy-tailed bases, smaller on smooth wavy ones) and the achieved statistic (`detect_z`, the weakest event of the series) are recorded per series, so difficulty can be stratified honestly. `CERTIFY_DETECTABILITY = False` restores uncertified magnitudes and durations.
- **Labelling guarantee**: every perturbation is multiplied by a trapezoidal window that is exactly zero at both segment boundaries, and the label covers the strictly-perturbed support (the zero-weight boundary samples are not labelled). The perturbed support therefore equals the labelled support exactly - a segment anomaly never leaves an unlabelled discontinuity outside its own label, nor an unmodified sample inside it.
- **Reproducibility**: one independent RNG per series, derived from `SEED` via `SeedSequence.spawn`. A series depends only on `(SEED, group, series_index)`, so pools are unchanged by pool size, generation order or parallelism. Each run writes `{group}_manifest.json` with the exact settings used.

## Unified Data Format

- **Canonical storage format**: Separate Parquet files per group, written to `data/01-anomaly-detection-agent-spec/` at the repository root - `real/` holds `R1.parquet` and `R2.parquet`, `synthetic/` holds `S1.parquet` and `S2.parquet`, each beside its `*_metadata.parquet`. Each file contains the entire pool for that group. Columns (all series flattened):
  - `series_id` (string, primary key)
  - `time_index` (int64) - monotonic integer starting at 0 **per series**
  - `value` (float64)
  - `label` (int8) - point-wise anomaly flag (0/1) from TSB-UAD

- **Series ID**: `{group}__{original_dataset}__{original_id}_{sample_id}`
  - `group`: `R1` / `R2` / `S1` / `S2`
  - `original_dataset`: e.g. `YAHOO`, `MITDB`, `SMD`
  - `original_id`: original identifier from source (e.g. `real_42`, `record_117`)
  - `sample_id`:
    - `"full"` - the original series was **not split**
    - `"chunk{N}"` - `N` is the 0-based chunk index (example: `R1__YAHOO__real_42_chunk0`)
    - `"clean{N}"` - legacy sampling only (see below); does not occur in the shipped pools

  For synthetic pools `original_dataset` is the literal `SYNTH` and `original_id` is `{base_type}-{index:05d}`, e.g. `S1__SYNTH__ar1-00042_full`, so the three-part `__` structure is identical to R1/R2.

  **Splitting rule (applied once during pool creation, see `src/sampler-r1-r2.py`)**:
  - If the length of the original series **L ≤ 15,000** (`SPLIT_THRESHOLD`) then set `sample_id = "full"`; the sample keeps the series' natural length.
  - If **L > 15,000** then cut non-overlapping chunks on a **fixed grid**. The chunk size is computed **once per series from the values alone**: it starts at 1,500 points (`BASE_CHUNK_SIZE`) and, when a dominant period is detected, is rounded up to a whole number of periods spanning at least 3 of them (`MIN_PERIODS_PER_CHUNK`). Every boundary then falls on a period multiple by construction. Periods come from a periodogram-validated autocorrelation detector (least-squares detrend plus a mild robust clip, candidate peaks refined on the ACF, harmonic comb scoring to pick the fundamental, prominence and correlation confirmation) - robust to anomalies and trend, and verified against a ground-truth suite (474/490 correct vs 352/490 for a plain first-ACF-peak scan; 1/18 vs 3/18 false alarms on aperiodic traps; 96% vs 64% agreement between the two halves of the same recording on R1).
  - Within one series all chunks share the same length; only the final chunk may be shorter (a function of the series length alone). A remainder below 1,000 points is dropped.
  - Every emitted sample satisfies **1,000 ≤ length ≤ 35,000** (`ABSOLUTE_MIN` / `ABSOLUTE_MAX`).
  - **Each point (and its `label`) is assigned to at most one sample** - chunks of one original series never overlap. Points may be dropped, never duplicated. `chunk{N}` numbers the grid slot, so a gap in the numbering marks a rejected slot.

  **Anomaly-ratio budget**: a positive series should *contain* an anomaly, not consist of one. A single cap `MAX_ANOMALY_RATIO = 0.30` is enforced **purely by rejection**: an over-budget slot is dropped, and the grid never moves in response. Above ~0.30 the label "series contains an anomaly" stops being an honest description and majority-based detectors start inverting; below the cap the natural per-chunk distribution is preserved untouched.

  **Representativeness screen (label-noise guard)**: a *clean-labelled* chunk wildly unlike its parent series is suspicious of unlabelled anomalies (known TSB-UAD label noise) and is dropped. The deviation of the chunk's raw mean/std is measured against a **robust reference** - the parent's statistics after winsorizing at 5 robust σ (`REFERENCE_CLIP_SIGMAS`), so a recording's own anomalies cannot widen its tolerance band - with threshold `STATS_TOLERANCE = 1.5`, placed where measured innocence ends: clean R1 chunks deviating 1.0-1.5 still look like the bulk (max excursion ≈ 3σ), while those beyond 1.5 carry unlabelled excursions of 9-44σ. The screen costs 4.2% of R1's clean slots and 0.7% of R2's.

  **Leak-freedom (the core design requirement)**: no sample boundary ever depends on the labels, so sample length cannot act as a proxy for `y_i`. The legacy logic (dilution toward a 0.06 target, anomaly-cluster expansion, whole-series triage, `clean{N}` salvage) stretched exactly the anomalous samples: a length-only classifier reached **0.97 AUROC on R2**, and 88.7% of dirty R2 samples sat pinned to a dilution threshold. Now the same probe scores 0.53 on R2 and 0.64 on R1 (the residual is the statistical fact that a longer window is likelier to contain an anomaly - the price of keeping realistic, period-honest per-series lengths), pinning is < 1%, and the mean length of positive vs negative samples is 1.00x on R2 (was 3.90x). The legacy behaviour remains available via `LEGACY_SAMPLER = True` and reproduces the old pools byte-for-byte.
  - Realised ratio maxima: R1 = 0.299, R2 = 0.299, S1 = 0.198, S2 = 0.299 (the S2 tail above 0.20 comes from the periodic duration floors - events on seasonal bases must span ≥ 2 full periods - which can exceed the drawn fraction on short series; everything stays under the 0.30 budget shared with the real pools).

  **Realised within-sample density** - with dilution gone, each pool keeps its domain's natural distribution instead of a value manufactured by the sampler:

  | Pool | p10 | median | p90 | max |
  |------|-----|--------|-----|-----|
  | R1 | 0.0020 | 0.0100 | 0.1880 | 0.2993 |
  | R2 | 0.0595 | 0.1253 | 0.2258 | 0.2992 |
  | S1 | 0.0094 | 0.0389 | 0.1277 | 0.1982 |
  | S2 | 0.0104 | 0.0627 | 0.1681 | 0.2985 |

  R1 and R2 sit an order of magnitude apart in density - that is a domain difference (IT-Ops point outliers vs. arrhythmic ECG), not a defect, and the synthetic pools span both. Density is therefore a **difficulty axis to stratify by in the report**, not a quantity to equalise.

  A sparse tail exists in R1: 9.7% of its positives carry ≤ 3 anomalous points. These are *not* label noise - measured against the local noise scale, their anomalous points deviate by a median of **≈ 8σ**: large, obvious spikes. Sparse-but-huge and dense-but-subtle are the two ends of the difficulty range, and both belong in the pool; no minimum-count filter is applied.

  **Pool composition (after chunking)** - the pool is assembled toward an explicit size by **water-filling**: one ceiling is shared by every recording and raised until the target is met, so recordings smaller than the ceiling contribute everything they have and larger ones contribute exactly the ceiling - the most balanced allocation that reaches the requested size. Every subsampling decision is a pure function of the sample's own identity (crc32) - never a label, a length or an RNG sequence - so no new leak can enter and rebuilds are stable. When the constraints conflict, the pool is built at the **largest size they all allow**:
  - **`TARGET_POOL_SIZE = 2900`** - the explicit size target (both pools land exactly on it).
  - **`TARGET_POSITIVE_RATE = 0.25`** - final share of `y_i = 1`, matched to the synthetic pools. Positives are water-filled first (the scarce class), negatives fill each recording's remaining allowance; in R2, 24 of the 30 recordings contribute positives (median 38 per recording).
  - **`MAX_RECORDING_SHARE = 0.05`** - no recording may exceed 5% of the final pool (chunks of one recording are near-duplicates; one LTDB record used to supply 36.2% of R2). The per-recording ceiling this implies is reported in the manifest (`recording_ceiling`).
  - `MAX_SAMPLES_PER_RECORDING` (default `None`) remains available as a fallback hard cap.
  - **Stratified grouped dev/test split** (`DEV_FRACTION = 0.30`): assigned per *machine-level identity* - raw files that are separate columns of one machine, trace or ECG record (`..._col_N`) count as one entity, since they share regimes and anomaly windows - so near-duplicate series can never straddle the evaluation boundary. Whole machines are placed in stable-hash order by a greedy pass plus deterministic local search, targeting `DEV_FRACTION` of both the *samples* and the *positives* (the grouped analogue of a stratified split). Realised: dev = 870/2,900 (30.0%) in both pools, dev/test positive rates 25.1%/25.0% (R1) and 24.0%/25.4% (R2). The split lands in the `split` metadata column; synthetic pools carry the same column, hashed per series.
  - **Full variants**: the same run also writes the uncomposed pools into a sibling `full/` directory - every valid sample, no size, rate or share shaping (`full/R1` = 4,082 at 34.7%, `full/R2` = 12,581 at 24.2%); the grouped split is still assigned (inherited from the composed pool's machine→side mapping), and `full/README.md` explains when to prefer them. `WRITE_FULL_POOLS = False` disables them. Synthetic pools have no full variant: they are generated directly to the requested size and rate, so nothing is ever dropped.
  - Synthetic pools run at `rate = 0.25` over 2,000 series: ~500 dirty series keep ~100 per family for per-family analysis, while ~1,500 clean series give detectors a serious false-positive surface.

- **Timestamps**: Absent. Only `time_index` (0, 1, 2, …) is used within each series. Missing values are handled according to TSB-UAD rules (forward-fill or drop).

- **Values**: `float64`, **raw** values from TSB-UAD (no additional scaling or unit conversion).

- **Metadata**: A separate small parquet file `{group}_metadata.parquet` (one per group) with the following columns:
  - `series_id`
  - `length`
  - `num_point_anomalies`
  - `y_i` (series-level label, see below)
  - `is_split` (boolean)
  - `original_length`
  - `source_notes` (free-form provenance tags: `period=…`; legacy pools may also carry `whole_series`, `clean_salvage`, `tail_merged`)

  All pools additionally carry:
  - `split` (`"dev"` / `"test"`) - evaluation split: a stratified grouped split per machine-level recording for real pools, hashed per series for synthetic ones

  Real pools (R1, R2) additionally carry:
  - `period_detected` (int or null) - dominant period, or null when none was found
  - `is_representative` (boolean) - whether a clean sample passed the representativeness screen (see above); screened-out samples are dropped, so in-pool rows always carry `true`
  - `anomaly_ratio` (float) - `num_point_anomalies / length`

  Synthetic pools (S1, S2) additionally carry:
  - `base_type`, `base_params` (JSON) - the generating process and its drawn parameters
  - `anomaly_type`, `severity`, `num_segments`, `segments` (JSON list of `{kind, start, end, severity, …}`)
  - `severity_effective` (float) - the size actually applied after the per-series adaptation, on the same dial as the drawn `severity` (multiples of σ_vis: the displacement for `point`/`level_shift`/`trend`, the peak-to-peak span for `group`/`seasonality`, and `2·(inflation-1)` for `variance`). It genuinely varies per series: above the drawn severity on wide-band or heavy-tailed bases, below it on smooth wavy ones
  - `detect_z` (float) - the certified detectability statistic of the series' weakest event (whitened matched-filter z; aggregated over the whole spike pattern for `point`)
  - `anomaly_fraction` (realised) and `target_fraction` (requested)

  The `segments` field gives exact ground-truth spans for the synthetic pools, which makes them usable for point-level evaluation as well as the series-level task defined below.

- **Integrity**: both scripts validate their output before writing it - unique `series_id`, contiguous `time_index` starting at 0, metadata agreeing with the actual labels, and the length and anomaly-ratio budgets. A violation aborts the run rather than producing a parquet file that later stages would trust.

## Label Definition (Series-Level)

### What is the prediction target?
- **Label type**: binary series-level label `y_i ∈ {0,1}`
- **Positive class definition**: anomalous time series (the series contains at least one anomalous point)
- **Negative class definition**: completely normal series
- **Ambiguous / unlabeled cases**: none (all series are assigned a `y_i`)

### How labels are derived (per dataset)
- **R1 derivation**: `y_i = 1` if the series (or chunk) contains at least one point with `label=1` (TSB-UAD point-wise labels). Otherwise, `y_i = 0`.
- **R2 derivation**: same rule applies (point-wise to series-level).
- **S1 derivation**: injected anomaly results in `y_i = 1`
- **S2 derivation**: injected anomaly results in `y_i = 1`

### Realised class balance
| Pool | Series | `y_i = 1` | dev / test |
|------|--------|-----------|------------|
| R1 | 2,900 | 25.0% | 870 / 2,030 |
| R2 | 2,900 | 25.0% | 870 / 2,030 |
| S1 | 2,000 | 26.2% | 595 / 1,405 |
| S2 | 2,000 | 24.2% | 596 / 1,404 |

### Known label noise / caveats
- R1 and R2: slight subjective noise may be present. Synthetic data has zero noise.
- When splitting long series, an anomaly will fall into only one chunk - this is expected and is correctly reflected in the `y_i` of each chunk.
- Dropping over-budget slots removes anomalous material preferentially, so the realised positive rate is lower than the raw share of dirty series in the source. This is intentional: it keeps positives interpretable as "contains an anomaly".
- An anomaly cluster falling on a grid boundary may split into two positive chunks, each holding part of the cluster. This is accepted: moving boundaries to avoid it would tie the grid to the labels, which is exactly the leak this design removes.

## Tool Suite
Enumerate tools and their deterministic I/O at the level needed for evaluation.

### Cheap scan (global)

- [TODO: enumerate all the cheap tools following the format below]
- **Function**: `cheap_scan_all(pool) -> cheap_scan.json`
- **Per-series outputs** (minimum):
  - `series_id`
  - `cheap_score` (higher = more anomalous)
- **Determinism rule**: same pool + config => identical output

### Deep detect (per series)

- [TODO: enumerate all the deep detect tools following the format below]
- **Function**: `deep_detect(series, preset) -> deep_detect.json`
- **Per-series outputs** (minimum):
  - `series_id`
  - `deep_score`
  - `diagnostics` (structured fields only; no freeform)
  - `evidence`: pointers to time indices / segments / summary stats used to justify the score
- **Determinism rule**: same series + preset + config => identical output

## Agent Contract (Planning + Guardrails)
Define exactly what the agent is allowed to do and what it must output.

### Inputs available to the agent
- Cheap scan results (`cheap_scan.json`)
- Optional per-series metadata (define fields): [TODO]

### Budget model
Define a budget `B` deep-dives per 1,000 series.

- **B definition**: number of series allowed to call `deep_detect` on
- **Budgets to evaluate**: [TODO: e.g., B ∈ {0, 10, 25, 50, 100}]
- **Hard enforcement**: agent run must fail (or truncate) if budget exceeded

### Agent outputs (must be machine-checkable)
- `ranked_list.json` with (minimum):
  - ordered list of `series_id`
  - final anomaly score used for ranking
  - recommended threshold policy (dynamic K)
- `decision_trace.json` with (minimum):
  - which series were deep-dived
  - justification referencing only tool outputs (IDs/fields)

### Faithfulness / evidence requirements
- All natural-language claims must be traceable to structured tool outputs
- No claims about raw series unless supported by evidence fields returned by tools
- If evidence is insufficient, agent must say so and defer

## Baselines (Definitions + Expected Outputs)
Baselines must produce the same `ranked_list.json` schema.

- **Baseline A: cheap-only**
  - Ranking uses only `cheap_score` (plus any allowed deterministic post-processing)
- **Baseline B: random deep-dive**
  - Randomly select B series for deep detect, then rank using a fixed rule

For each baseline:
- **Tie-breaking rule**: [TODO]
- **Randomness control**: [TODO: seed policy]

## Metrics and Benchmark Reporting
Define metrics, how they’re computed, and what plots/tables are required.

### Ranking metrics
- AUROC for the varying threshold
- Adjusted Rand Index for the recommended threshold

### Budget / cost accounting
- Tool-call counts (cheap + deep)
- Runtime per tool and total runtime
- Quality-vs-budget curves: metric vs B

### Required benchmark report artifacts per run
- `metrics.json`
- single aggregated run summary (human-readable markdown): [TODO: filename]
