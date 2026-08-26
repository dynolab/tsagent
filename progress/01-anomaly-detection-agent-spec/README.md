# 01 - Anomaly Detection Agent Spec

Defines the task and builds the four evaluation pools it will run on. The full contract
lives in [`spec.md`](spec.md); this file covers what is done and how to rebuild the data.

## Status

- [x] Fill "TODO" fields left in `spec.md` - dataset-side sections are complete. The
      agent-side TODOs (tool suite, budget model, baseline tie-breaking, metrics artefacts)
      are deliberately still open; they belong to the next task.
- [x] Find 2 real datasets (about 1k time series) for anomaly detection - **R1** (IT-Ops)
      and **R2** (biomedical), sampled from TSB-UAD-Public-v2.
- [x] Build 2 synthetic datasets specifying the base processes and anomaly types -
      **S1** (stationary) and **S2** (trend-seasonal).

## Pools

| Pool | Source | Series | `y_i = 1` | dev / test | Max anomaly ratio |
|------|--------|--------|-----------|------------|-------------------|
| R1 | YAHOO, SMD, IOPS, Exathlon, WSD, NEK | 2,900 | 25.0% | 870 / 2,030 | 0.299 |
| R2 | MITDB, SVDB, LTDB | 2,900 | 25.0% | 870 / 2,030 | 0.299 |
| S1 | `white_noise`, `ar1`, `ar2` | 2,000 | 26.2% | 595 / 1,405 | 0.198 |
| S2 | `linear_trend`, `seasonal_sine`, `trend_seasonal` | 2,000 | 24.2% | 596 / 1,404 | 0.299 |

9,800 series in total, every pool at a ~25% positive rate. Alongside the composed real
pools, `full/R1.parquet` (4,082 samples, 34.7% positive) and `full/R2.parquet` (12,581,
24.2%) hold every valid sample with no size, rate or share composition, for experiments
that want the raw yield. Each pool is a pair of parquet files -
`{pool}.parquet` (`series_id`, `time_index`, `value`, `label`) and
`{pool}_metadata.parquet` (one row per series, `y_i` among the columns) - written under
`data/01-anomaly-detection-agent-spec/` at the repository root:

```
data/01-anomaly-detection-agent-spec/
  real/       R1, R2 + metadata + manifests   <- the pools to work with
  synthetic/  S1, S2 + metadata + manifests
  full/       R1, R2 + metadata + README      <- uncomposed: every sample the
                                                 sampler produced, no size/rate/
                                                 share shaping (see its README)
```

## Layout

```
spec.md                          task contract, data format, label definition
src/sampler-r1-r2.py             builds R1 + R2 from raw_data/
src/generator-s1-s2.py           builds S1 + S2 from scratch
notebooks/raw_real_datasets_analysis.ipynb   exploration of the raw TSB-UAD files
notebooks/real_datasets_analysis.ipynb       browse the built R1/R2 pools
notebooks/synthetic_datasets_analysis.ipynb  browse the built S1/S2 pools
(pools land in ../../data/01-anomaly-detection-agent-spec/, git-ignored)
```

## Rebuilding the data

Both scripts are configuration-driven: every knob sits in the `CONFIGURATION` block at the
top of the file, and there are no command-line arguments. Edit the block, run the file.

Install the dependencies once:

```bash
pip install -r ../../code/requirements.txt
```

Real pools - expects `raw_data/R1/<dataset>/*.csv|parquet` next to the repository, i.e.
`PROJECT/raw_data/` (set by `RAW_DATA_DIR`). Takes a few minutes:

```bash
python src/sampler-r1-r2.py
```

Synthetic pools - no input data needed, runs in a few seconds:

```bash
python src/generator-s1-s2.py
```

The sampler writes into `data/<task>/real/`, the generator into `data/<task>/synthetic/`.
Both validate the result before writing: unique `series_id`,
contiguous `time_index`, metadata agreeing with the labels, and the length and
anomaly-ratio budgets. A violated invariant aborts the run instead of producing a file
later stages would trust.

`generator-s1-s2.py` also writes `{pool}_manifest.json` with the exact settings used, and
is reproducible to the byte: a series depends only on `(SEED, pool, index)`, not on
pool size or generation order.

## Design decisions worth knowing

**Leak-free sampling.** No chunk boundary ever depends on the labels: the chunk size
is computed once per series from the values alone (period-aware, so real series keep
their different natural lengths), chunks sit on a fixed grid, and the anomaly budget
(cap 0.30) is enforced purely by rejecting over-budget slots. Under the legacy logic
- dilution and cluster expansion stretched exactly the dirty samples - a length-only
classifier scored 0.97 AUROC on R2; now the same probe scores 0.53 on R2 and 0.64 on
R1 (the residual is just "a longer window is likelier to contain an anomaly") and
positive/negative mean lengths differ by 1.00x on R2 instead of 3.90x. Set
`LEGACY_SAMPLER = True` in the sampler to reproduce the old pools byte-for-byte.

**Disjoint coverage.** Chunks of one original series never overlap. Points may be dropped
when a slot busts the budget, but never duplicated - so a series-level metric cannot be
inflated by the same data appearing twice.

**Water-filling pool composition.** After chunking, each pool is assembled to an explicit
size (`TARGET_POOL_SIZE = 2900`) at a 25% positive rate by water-filling: one ceiling
shared by all recordings rises until the target is met, so small recordings contribute
everything and no recording exceeds 5% of the pool (`MAX_RECORDING_SHARE`; one LTDB
record used to supply 36% of R2). If the constraints conflict, the pool is built at the
largest size they all allow. A stratified grouped dev/test split (30/70, whole machines
placed by a deterministic greedy-plus-local-search pass targeting 30% of both the samples
and the positives, `split` column) keeps near-duplicate series on one side of the
evaluation boundary - realised dev/test positive rates are 25.1%/25.0% on R1 and
24.0%/25.4% on R2. Which samples of a recording take its slots stays a crc32 function of
the sample's own identity: deterministic, order-invariant, and blind to values and
lengths.

**Robust period detection and a robust label-noise screen.** Chunk sizes follow periods
found by a periodogram-validated autocorrelation detector (least-squares detrend, mild
robust clipping, harmonic comb scoring) - on a ground-truth suite it is correct in
474/490 cases vs 352/490 for a plain first-ACF-peak scan, and its detections agree
between the two halves of the same recording 96% of the time vs 64%. Clean-labelled
chunks wildly unlike their parent recording (mean/std beyond 1.5 robust reference σ) are
dropped as suspected unlabelled anomalies; the reference is winsorized so a recording's
own anomalies cannot widen its tolerance band.

**Enriched synthetic regime.** Heavy-tailed (Student-t) innovations on ~25% of
series, AR(1)-coloured noise under trend/seasonal bases, nested second seasonality, slow
amplitude modulation, random baseline offsets, a `flatline` (stuck-sensor) anomaly family,
and two disjoint events in ~25% of dirty series. Heavy-tail innovations are redrawn
beyond 3.5σ: tails mean frequent 2.5-3.5σ shoulders - hard negatives - not lone
8-29σ monsters; a clean series must never out-spike the labelled anomalies (worst clean
excursion dropped from 29σ_vis to 6.1, matching what same-length Gaussian series show).

**Adaptive visible contrast.** Anomalies are sized against the visible geometry of the
particular series: its rolling-median baseline, the robust band around it
(`sigma_vis`), the reach of its own extremes (q99.5 of the residual) and the residual's
tail step (q99.9 - q99.5). Segment families displace by `severity × sigma_vis`
(peak-to-peak for symmetric waveforms), and the adaptation is two-sided - floored
against the masking the series itself produces, capped when the drawn size would dwarf
what the series itself does (`group` amplitude within [½ reach, ½ reach + ¾σ_vis];
`level_shift`/`trend` capped just above the reach, floored against the baseline's own
wander; `variance` above the natural rolling-scale swing; warped `seasonality`
calibrated up *or* down to span `severity × sigma_vis` peak-to-peak). `point` spikes
land at `baseline ± d`, `d = reach + margin(severity, tail)` - always beyond the
series' own excursions, anchored to the baseline (added on top of the sample, a +3σ
spike on a -3σ sample lands at the mean), with an edge keep-out and per-spike jitter.
Certification only raises: every event must additionally reach a whitened
matched-filter z of 5 against the series' known AR polynomial (for `point`, aggregated
over the whole pattern), and duration rules keep degenerate combos out (sparse `point`
density, ≥ 2 full periods for steps/drifts on periodic bases, ≥ 1 full cycle of
discrepancy for a warp, per-family duration floors). The drawn `severity` stays the
difficulty axis; the size actually applied (`severity_effective`) and the achieved
statistic (`detect_z`) land in the metadata for honest difficulty stratification.
Validated on every dirty series: the weakest labelled spike sits at ≥ ~0.9 of its
series' own reach in every base × noise combination, and the count of clean samples
protruding past the weakest labelled spike dropped from hundreds to 0-3 (median).

**Log-uniform anomaly density.** The synthetic anomalous-point fraction is drawn
log-uniformly from [0.01, 0.20], matching the two orders of magnitude the real pools
span (R1 median ≈ 0.010, R2 median ≈ 0.125) and giving quality-vs-density curves support
along the whole axis.

**Severity in visible sigma.** Synthetic anomaly magnitudes are measured in `sigma_vis`
- the robust spread of the clean series around a slow rolling-median baseline (window
wider than the seasonal period). Trend is excluded from the unit, the seasonal swing is
included, so severity 3.0 means "three of the bands this series itself shows" on
`white_noise` and on `trend_seasonal` alike. `severity_effective` reports the size the
adaptation actually applied on the same dial - larger than drawn on wide-band or
heavy-tailed series, smaller on smooth wavy ones.

**Browsing the pools.** Both notebooks share a small helper layer: `load_pool(name, kind)`
returns a pool and its metadata, and `show(pool, meta, ...)` selects series and plots them.
Every filter is optional and composable - `anomaly_type`, `base_type`, `dataset`, `split`,
`anomalous`, `min_ratio`/`max_ratio`, `sort_by`, `random_state`, plus `start`/`end` to zoom.
So `show(s1, s1_meta, n=3, anomaly_type="flatline")` or
`show(r1, r1_meta, n=3, max_ratio=0.005)` is the whole workflow.

**Honest labels.** Every synthetic perturbation is tapered to exactly zero at both segment
boundaries, and the label covers the strictly-perturbed support only - no unlabelled
discontinuity outside the span, no unmodified sample inside it.
