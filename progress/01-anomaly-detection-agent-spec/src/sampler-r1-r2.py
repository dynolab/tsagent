"""Time-series sampler that turns the raw TSB-UAD pools into the R1/R2 pools of spec.md.

Reads the raw series of a group, splits long ones into period-aligned chunks, keeps the
share of anomalous points inside each sample within budget, and writes ``{group}.parquet``
plus ``{group}_metadata.parquet`` in the unified format.

Every tunable knob lives in the CONFIGURATION block below.

The previous behaviour is kept intact behind ``LEGACY_SAMPLER = True`` for byte-exact
comparison runs; everything below describes the current logic.

Guarantees (both modes)
-----------------------
Disjoint coverage
    Chunks of one series never overlap; a point belongs to at most one sample.

Unique keys
    ``series_id`` is unique across the pool.

Length budget
    Every sample satisfies ``ABSOLUTE_MIN <= length <= ABSOLUTE_MAX``.

Leak-free sampling
------------------
No sample boundary ever depends on the labels: the chunk size is computed once per
series from the values alone (period-aware, so real series keep their different natural
sizes), chunks are laid on a fixed grid, and the anomaly budget
(``MAX_ANOMALY_RATIO``) is enforced purely by rejecting over-budget slots. Sample
length therefore cannot act as a proxy for ``y_i``. Under the legacy logic, whose
dilution and cluster expansion stretched exactly the dirty samples, a length-only
classifier reached 0.97 AUROC on R2; that is the leak this design closes.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================================
# CONFIGURATION - every knob for sampling lives in this block
# ======================================================================================

# All paths resolve from this file, so the script behaves the same whatever directory it
# is launched from and follows a rename of the task directory.
_HERE = Path(__file__).resolve()
_TASK_DATA = _HERE.parents[3] / "data" / _HERE.parents[1].name

RAW_DATA_DIR = _HERE.parents[4] / "raw_data"   # input: raw pools, kept outside the repo
OUTPUT_DIR = _TASK_DATA / "real"               # composed pools - the ones to work with
FULL_OUTPUT_DIR = _TASK_DATA / "full"          # uncomposed pools, see WRITE_FULL_POOLS

# Which pools to build, and the sub-directory of RAW_DATA_DIR each one reads.
GROUPS_TO_BUILD: Dict[str, str] = {"R1": "R1", "R2": "R2"}

# --- Chunking ---
SPLIT_THRESHOLD = 15000     # series longer than this are split into chunks
BASE_CHUNK_SIZE = 1500      # starting chunk size, grown to whole periods when one is
                            # detected. This is the knob that shapes the pool: it sets
                            # how much context a sample carries and, with it, the natural
                            # positive rate (a longer window is likelier to contain an
                            # anomaly). Measured on this pipeline, the natural rate
                            # at base 1000/1500/2000 runs 27/35/40% on R1 and 18/24/29%
                            # on R2 - so 1500 keeps R2 essentially on the 25% target
                            # while R1's surplus positives (which it has in abundance)
                            # are trimmed by composition; 1000 would halve the context
                            # of aperiodic series only to starve R2 of positives.
ABSOLUTE_MIN = 1000         # hard lower bound on an emitted sample
ABSOLUTE_MAX = 35000        # hard upper bound on an emitted sample and the only upper
                            # bound - a chunk grows only to span whole periods

# --- Legacy mode ---
# False (default): leak-free logic as described in the module docstring.
# True: the previous behaviour - dilution toward LEGACY_TARGET_ANOMALY_RATIO,
#   anomaly-cluster expansion, whole-series triage and clean salvage. Kept byte-exact
#   for comparison runs; nothing else needs to change to switch.
LEGACY_SAMPLER = False

# --- Anomaly budget (share of anomalous points inside one sample) ---
# A single rejection cap. Above it, "series contains an anomaly" stops being an honest
# description and majority-based detectors start inverting.
MAX_ANOMALY_RATIO = 0.30

# Legacy-only knobs (ignored otherwise):
LEGACY_TARGET_CHUNK_MAX = 8000            # soft chunk cap; the current grid drops it -
                                          # defined as max(cap, period * MIN_PERIODS_PER_CHUNK),
                                          # it raised itself to whatever the period demanded
LEGACY_TARGET_ANOMALY_RATIO = 0.06        # dilution aims here first
LEGACY_ACCEPTABLE_ANOMALY_RATIO = 0.15    # fallback bound; every emitted sample satisfies it
LEGACY_MAX_ANOMALY_RATIO = 0.27           # above this a whole series is treated as degenerate

# --- Anomaly cluster handling ---
ANOMALY_LOOKAHEAD = 100     # gap tolerated when joining anomalies into one cluster

# --- Representativeness screen (clean samples only) ---
# A clean-labelled chunk wildly unlike its parent series is suspicious of unlabelled
# anomalies (known TSB-UAD label noise) and is dropped. The deviation is measured
# against a ROBUST reference - the parent's mean/std after winsorizing at
# REFERENCE_CLIP_SIGMAS robust sigmas - so a recording's own anomalies cannot widen
# its tolerance band. The threshold sits where measured innocence ends: on R1, clean
# chunks deviating 1.0-1.5 still look like the bulk (max excursion ~3 sigma, vs 2.1 for
# the crowd), while those beyond 1.5 carry unlabelled excursions of 9-44 sigma. Cost:
# 4.2% of R1's clean slots, 0.7% of R2's; the screen errs toward label purity.
STATS_TOLERANCE = 1.5
REFERENCE_CLIP_SIGMAS = 5.0

# Legacy-only: plain mean/std reference with a much tighter band. At 0.65 it rejected
# 38% of R1's clean chunks - ordinary quiet/busy regimes, i.e. the hard negatives - and
# pushed the natural positive rate from 28.7% to 39.5%. Kept for byte-exact legacy runs.
LEGACY_STATS_TOLERANCE = 0.65

# --- Pool composition (set a control to None to disable it) ---
# The pool is assembled toward an explicit size target by WATER-FILLING: one ceiling is
# shared by every recording and raised until the target is met, so recordings smaller
# than the ceiling contribute everything they have and larger ones contribute exactly
# the ceiling - the most balanced allocation that reaches the requested size. Every
# subsampling decision is a keyed hash of the sample's own identity, never a label, a
# length or an RNG sequence - deterministic, order-invariant, and free of new leaks.
# When the constraints conflict, the pool is built at the LARGEST size they all allow.
TARGET_POOL_SIZE = 2900         # final samples per pool; None = as many as the other
                                # constraints permit
TARGET_POSITIVE_RATE = 0.25     # final share of y_i=1 series, matched to the synthetic
                                # pools; None = keep the natural rate
MAX_RECORDING_SHARE = 0.05      # no recording may exceed this share of the final pool:
                                # chunks of one recording are near-duplicates, and an
                                # unbounded recording dominates the metric (one LTDB
                                # record used to supply 36% of R2); None disables
MAX_SAMPLES_PER_RECORDING = None  # optional hard per-recording cap, kept as a fallback;
                                  # the share-based control above supersedes it
COMPOSITION_SEED = 42           # keys every hash-based selection above
WRITE_FULL_POOLS = True         # also write the uncomposed pools into FULL_OUTPUT_DIR:
                                # every valid sample, no size/rate/share shaping (the
                                # grouped dev/test split is still assigned)

# --- Grouped evaluation split ---
# Assigned per MACHINE-LEVEL RECORDING, so chunks of one recording can never straddle
# dev/test - a random per-sample split would leak near-duplicate chunks across the
# boundary. Whole machines are walked in stable-hash order and placed in dev whenever
# that brings the dev side closer to BOTH quotas - the sample share and the positive
# share (a stratified grouped split). A bare hash of the identity lands wherever
# recording sizes happen to fall: it once put 13% of R1 in dev with 42% positives after
# period-aware chunking changed the per-machine yields.
DEV_FRACTION = 0.30             # share of samples targeted for "dev"; rest is "test"

# --- Period detection ---
MIN_PERIOD_DETECT = 25
MAX_PERIOD_DETECT = 5000
MIN_PERIODS_PER_CHUNK = 3   
MIN_LENGTH_FOR_PERIOD = 150 
PERIOD_CLIP_SIGMAS = 8.0 # pre-clip before spectral analysis: mild winsorizing so
                            # isolated anomalies cannot wreck the detrend or the
                            # spectrum, while genuine waveform peaks (ECG QRS) survive
# ======================================================================================

def recording_group(orig_id: str) -> str:
    """Identity of the physical source behind a raw file.

    Several raw files are separate *columns* of one machine, trace or ECG record
    (``machine-1-1_col_11``, ``215_col_0`` / ``215_col_1``); their series share regimes
    and anomaly windows, so for split purposes they are one entity. Dropping the
    ``_col_N`` suffix groups them; files without the suffix are their own group.
    """
    return re.sub(r"_col_\d+$", "", orig_id)


def stable_rank(*parts: object) -> float:
    """Deterministic pseudo-uniform value in [0, 1) keyed by ``parts``.

    Built on crc32 rather than ``hash()`` (randomised per interpreter) or an RNG sequence
    (order-dependent): the decision for one sample depends only on its own identity, so
    adding or removing other recordings never reshuffles the selection.
    """
    key = "|".join(str(p) for p in parts).encode()
    return (zlib.crc32(key) & 0xFFFFFFFF) / 2**32


def robust_reference(values: np.ndarray) -> Dict[str, float]:
    """Label-free reference statistics for the representativeness screen.

    The plain mean/std of a series are inflated by its own anomalies, which hands
    exactly the dirtiest recordings the widest tolerance bands. Winsorizing at
    REFERENCE_CLIP_SIGMAS robust sigmas strips that influence; the chunk under test
    keeps its raw statistics, so unlabelled anomalies inside a "clean" chunk are what
    push it over the line - the point of the screen. A series whose MAD collapses to
    zero (integer counts with long constant runs) is used as-is: clipping everything to
    the median would reject every chunk of it.
    """
    med = float(np.median(values))
    mad_sigma = 1.4826 * float(np.median(np.abs(values - med)))
    if mad_sigma > 1e-10:
        lim = REFERENCE_CLIP_SIGMAS * mad_sigma
        w = np.clip(values, med - lim, med + lim)
    else:
        w = values
    return {"mean": float(np.mean(w)), "std": float(np.std(w))}


@dataclass
class SampleRecord:
    """A single sampled time-series segment in the unified format."""

    series_id: str
    time_index: np.ndarray
    value: np.ndarray
    label: np.ndarray
    length: int
    num_point_anomalies: int
    y_i: int
    is_split: bool
    original_length: int
    source_notes: Optional[str] = None
    period_detected: Optional[int] = None
    is_representative: bool = True
    anomaly_ratio: float = 0.0
    # Absolute bounds inside the original series; the chunk loop resumes from _end_idx,
    # which is what keeps consecutive chunks disjoint.
    _start_idx: int = field(default=0, repr=False)
    _end_idx: int = field(default=0, repr=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Flat frame for parquet storage."""
        return pd.DataFrame(
            {
                "series_id": self.series_id,
                "time_index": self.time_index,
                "value": self.value.astype(np.float64),
                "label": self.label.astype(np.int8),
            }
        )

    def to_metadata_row(self) -> Dict:
        """Metadata row for ``{group}_metadata.parquet``."""
        return {
            "series_id": self.series_id,
            "length": self.length,
            "num_point_anomalies": self.num_point_anomalies,
            "y_i": self.y_i,
            "is_split": self.is_split,
            "original_length": self.original_length,
            "source_notes": self.source_notes or "",
            "period_detected": self.period_detected,
            "is_representative": self.is_representative,
            "anomaly_ratio": round(self.anomaly_ratio, 4),
        }


class TimeSeriesSampler:
    """Adaptive sampler with period-aware chunking and an explicit anomaly budget."""

    def __init__(
        self,
        split_threshold: int = SPLIT_THRESHOLD,
        base_chunk_size: int = BASE_CHUNK_SIZE,
        legacy_target_chunk_max: int = LEGACY_TARGET_CHUNK_MAX,
        absolute_min: int = ABSOLUTE_MIN,
        absolute_max: int = ABSOLUTE_MAX,
        legacy_target_anomaly_ratio: float = LEGACY_TARGET_ANOMALY_RATIO,
        legacy_acceptable_anomaly_ratio: float = LEGACY_ACCEPTABLE_ANOMALY_RATIO,
        legacy_max_anomaly_ratio: float = LEGACY_MAX_ANOMALY_RATIO,
        anomaly_lookahead: int = ANOMALY_LOOKAHEAD,
        legacy_stats_tolerance: float = LEGACY_STATS_TOLERANCE,
        stats_tolerance: float = STATS_TOLERANCE,
        min_period_detect: int = MIN_PERIOD_DETECT,
        max_period_detect: int = MAX_PERIOD_DETECT,
        legacy: bool = LEGACY_SAMPLER,
        max_anomaly_ratio: float = MAX_ANOMALY_RATIO,
    ) -> None:
        if not 0 < max_anomaly_ratio < 1:
            raise ValueError(f"max_anomaly_ratio must lie in (0, 1), got {max_anomaly_ratio}")
        if stats_tolerance <= 0:
            raise ValueError(f"stats_tolerance must be positive, got {stats_tolerance}")
        if not 0 < legacy_target_anomaly_ratio <= legacy_acceptable_anomaly_ratio <= legacy_max_anomaly_ratio < 1:
            raise ValueError(
                "anomaly ratios must satisfy 0 < target <= acceptable <= max < 1, got "
                f"{legacy_target_anomaly_ratio}, {legacy_acceptable_anomaly_ratio}, {legacy_max_anomaly_ratio}"
            )
        if not 0 < absolute_min <= base_chunk_size <= absolute_max:
            raise ValueError(
                "chunk sizes must satisfy 0 < absolute_min <= base_chunk_size <= absolute_max"
            )
        if legacy and not base_chunk_size <= legacy_target_chunk_max <= absolute_max:
            raise ValueError(
                "legacy mode requires base_chunk_size <= legacy_target_chunk_max <= absolute_max"
            )

        self.split_threshold = split_threshold
        self.base_chunk_size = base_chunk_size
        self.legacy_target_chunk_max = legacy_target_chunk_max
        self.absolute_min = absolute_min
        self.absolute_max = absolute_max
        self.legacy_target_anomaly_ratio = legacy_target_anomaly_ratio
        self.legacy_acceptable_anomaly_ratio = legacy_acceptable_anomaly_ratio
        self.legacy_max_anomaly_ratio = legacy_max_anomaly_ratio
        self.anomaly_lookahead = anomaly_lookahead
        self.legacy_stats_tolerance = legacy_stats_tolerance
        self.stats_tolerance = stats_tolerance
        self.min_period_detect = min_period_detect
        self.max_period_detect = max_period_detect
        self.legacy = legacy
        self.max_anomaly_ratio = max_anomaly_ratio

        self.rejected_count = 0     # total dropped samples (all causes)
        self.rejected_budget = 0    # dropped for exceeding the anomaly budget
        self.rejected_repr = 0      # dropped by the representativeness screen
        self.degenerate_count = 0   # series salvaged for clean stretches (legacy only)

    # ----------------------------------------------------------------------------------
    # Signal analysis
    # ----------------------------------------------------------------------------------

    def detect_period(self, values: np.ndarray) -> Optional[int]:
        """Dominant period of a series, from the values alone.

        The legacy mode keeps its single-pass ACF scan byte-for-byte; otherwise the
        periodogram-plus-ACF detector below is used, which is robust to anomalies and
        trend (the legacy detrend through the two literal endpoint values collapses
        under a single spike at either end: 175/200 -> 8/200 correct on a synthetic
        ground-truth suite).
        """
        if self.legacy:
            return self._detect_period_legacy(values)
        return self._detect_period_robust(values)

    def _detect_period_legacy(self, values: np.ndarray) -> Optional[int]:
        """Legacy detector: first ACF peak after an endpoint-anchored detrend."""
        n = values.size
        max_p = min(self.max_period_detect, n // MIN_PERIODS_PER_CHUNK)
        if max_p < self.min_period_detect:
            return None

        v = values - np.linspace(values[0], values[-1], n)
        std = np.std(v)
        if std < 1e-10:
            return None
        v = (v - np.mean(v)) / std

        f = np.fft.rfft(v, n=2 * n)
        acf = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n]
        acf = acf / (acf[0] + 1e-10)

        threshold = max(0.2, 1.96 / np.sqrt(n))
        search_limit = min(max_p, n // 2 - 1)
        for lag in range(self.min_period_detect, max(self.min_period_detect + 1, search_limit)):
            if acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1] and acf[lag] > threshold:
                # Confirm on the raw series, else detrending artefacts read as periods.
                with np.errstate(invalid="ignore"):
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                if np.isfinite(corr) and corr >= 0.15:
                    return lag
        return None

    def _detect_period_robust(self, values: np.ndarray) -> Optional[int]:
        """Periodogram candidates validated and refined on the autocorrelation.

        Pipeline (all label-free): least-squares detrend and a mild robust clip, so a
        handful of anomalies cannot wreck either step; candidate periods from the top
        periodogram peaks and the top ACF peaks; each candidate refined to the local
        ACF maximum and required to be a genuine hill (prominence over the preceding
        trough - a slowly wandering series keeps its ACF uniformly high and must not
        read as periodic); candidates ranked by a comb score over harmonic lags k*p,
        which picks the fundamental over its multiples; a final autocorrelation check
        at the chosen lag confirms on the detrended series.
        """
        n = values.size
        max_p = min(self.max_period_detect, n // MIN_PERIODS_PER_CHUNK)
        if max_p < self.min_period_detect:
            return None

        t = np.arange(n, dtype=np.float64)
        x = values.astype(np.float64, copy=False)
        slope, intercept = np.polyfit(t, x, 1)
        v = x - (slope * t + intercept)
        med = np.median(v)
        mad_sigma = 1.4826 * np.median(np.abs(v - med))
        if mad_sigma > 1e-10:
            lim = PERIOD_CLIP_SIGMAS * mad_sigma
            v = np.clip(v, med - lim, med + lim)
        std = np.std(v)
        if std < 1e-10:
            return None
        v = (v - np.mean(v)) / std

        nfft = 2 * n
        f = np.fft.rfft(v, n=nfft)
        power = np.abs(f) ** 2
        acf = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
        acf = acf / (acf[0] + 1e-10)
        # undo the taper of the biased estimator so peaks at long lags are comparable
        acf = acf * (n / np.maximum(n - np.arange(n), 1))

        search_hi = min(max_p, n // 2 - 2)
        if search_hi <= self.min_period_detect:
            return None

        candidates: set = set()

        k_lo = max(1, int(np.ceil(nfft / search_hi)))
        k_hi = min(power.size - 2, int(np.floor(nfft / self.min_period_detect)))
        if k_hi > k_lo:
            band = power[k_lo:k_hi + 1]
            floor_p = np.median(band) + 1e-30
            local = (band[1:-1] > band[:-2]) & (band[1:-1] > band[2:]) & (band[1:-1] > 10 * floor_p)
            peak_idx = np.flatnonzero(local) + 1
            for i in peak_idx[np.argsort(band[peak_idx])[::-1][:5]]:
                candidates.add(nfft / (k_lo + i))

        seg = acf[self.min_period_detect:search_hi]
        local = (seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:]) & (seg[1:-1] > 0.2)
        peak_idx = np.flatnonzero(local) + 1 + self.min_period_detect
        for i in peak_idx[np.argsort(acf[peak_idx])[::-1][:5]]:
            candidates.add(float(i))

        if not candidates:
            return None

        def refined(p0: float) -> Optional[int]:
            w = max(2, int(round(p0 * 0.06)))
            lo = max(self.min_period_detect, int(round(p0)) - w)
            hi = min(search_hi, int(round(p0)) + w)
            if hi <= lo:
                return None
            return lo + int(np.argmax(acf[lo:hi + 1]))

        def comb_score(p: int) -> float:
            k_max = max(1, min(4, (n // 2 - 2) // p))
            vals = []
            for k in range(1, k_max + 1):
                w = max(2, p // 20)
                lo, hi = k * p - w, min(n - 1, k * p + w)
                vals.append(float(np.max(acf[lo:hi + 1])))
            return float(np.mean(vals))

        scored: Dict[int, float] = {}
        for p0 in candidates:
            p = refined(p0)
            if p is None or p in scored:
                continue
            height = float(acf[p])
            if height < 0.2:
                continue
            trough = float(np.min(acf[1:p]))
            if height - trough < 0.1:
                continue
            scored[p] = comb_score(p)

        if not scored:
            return None
        best_score = max(scored.values())
        # Among near-equal scores prefer the smallest period: the fundamental. The band
        # is symmetric around the maximum via abs(), so it stays non-empty when every
        # comb score is negative (a bare 0.95 factor would then exclude the maximum).
        band = 0.05 * abs(best_score)
        best_p = min(p for p, sc in scored.items() if sc >= best_score - band)

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(v[:-best_p], v[best_p:])[0, 1]
        if not (np.isfinite(corr) and corr >= 0.2):
            return None
        return int(best_p)

    def compute_optimal_chunk_size(self, period: Optional[int]) -> int:
        """Chunk size: start from the base size, round up to whole periods.

        Depends on the values only (via the detected period), never on the labels - this
        is the sole source of length variation between series, which is exactly the
        "real series come in different sizes" property worth keeping.

        A chunk must span at least MIN_PERIODS_PER_CHUNK whole periods, otherwise it
        cannot show what "normal" repetition looks like; that requirement is what may
        push the size above BASE_CHUNK_SIZE, and ABSOLUTE_MAX is the only bound above it.
        """
        size = self.base_chunk_size
        if period and period >= self.min_period_detect:
            size = max(size, period * MIN_PERIODS_PER_CHUNK)
            size = ((size + period - 1) // period) * period
        if self.legacy:
            # The legacy soft cap could be exceeded only to fit a single period.
            size = min(size, max(self.legacy_target_chunk_max, period or 0))
        return int(np.clip(size, self.absolute_min, self.absolute_max))

    def align_boundary(self, pos: int, period: Optional[int], length: int, is_start: bool) -> int:
        """Snap a position to a period boundary: down for a start, up for an end."""
        if not period or period < self.min_period_detect:
            return pos
        if is_start:
            return max(0, (pos // period) * period)
        return min(length, ((pos + period - 1) // period) * period)

    def expand_anomaly_cluster(
        self, labels: np.ndarray, start: int, end: int, period: Optional[int] = None
    ) -> Tuple[int, int]:
        """Push ``end`` forward so a cluster of nearby anomalies is not cut in half.

        Capped at ``absolute_max`` from ``start``; an unbounded walk would swallow the
        rest of the series.
        """
        cap = min(labels.size, start + self.absolute_max)
        if end >= cap:
            return start, cap

        anomalies = np.flatnonzero(labels[start:end] == 1)
        if anomalies.size == 0:
            return start, end

        pos = start + int(anomalies[-1])
        new_end = end
        while True:
            window_start = pos + 1
            window_end = min(cap, window_start + self.anomaly_lookahead)
            if window_start >= window_end:
                break
            hits = np.flatnonzero(labels[window_start:window_end] == 1)
            if hits.size == 0:
                break
            pos = window_start + int(hits[-1])
            new_end = pos + 1

        new_end = self.align_boundary(new_end, period, labels.size, is_start=False)
        return start, min(new_end, cap)

    def check_representativeness(self, chunk: np.ndarray, g_mean: float, g_std: float) -> bool:
        """Whether a chunk's mean and spread are close enough to the series reference.

        Both tolerances are in units of the reference spread. Scaling the mean tolerance
        by ``|g_mean|`` instead would collapse the band to nothing for a series centred
        near zero, rejecting almost every clean chunk of it. The reference is robust
        (:func:`robust_reference`) with band STATS_TOLERANCE; the legacy mode keeps its
        plain-stats form and tighter band.
        """
        tol = self.stats_tolerance if not self.legacy else self.legacy_stats_tolerance
        scale = max(g_std, 1e-10)
        mean_ok = abs(float(np.mean(chunk)) - g_mean) <= tol * scale
        std_ok = abs(float(np.std(chunk)) - g_std) <= tol * scale
        return bool(mean_ok and std_ok)

    @staticmethod
    def _recalculate_label(labels: np.ndarray) -> Tuple[int, int, float]:
        """Series-level label, anomaly count and ratio from point-wise labels."""
        count = int(np.sum(labels))
        length = labels.size
        return (1 if count > 0 else 0), count, (count / length if length else 0.0)

    # ----------------------------------------------------------------------------------
    # Dilution
    # ----------------------------------------------------------------------------------

    def _optimize_expansion(
        self, labels: np.ndarray, start: int, end: int, total_len: int
    ) -> int:
        """Smallest forward expansion that brings the anomaly ratio down.

        Aims for ``legacy_target_anomaly_ratio``, falling back to ``legacy_acceptable_anomaly_ratio``
        only when the target is unreachable. Exact prefix-sum sweep: the ratio is not
        monotonic in ``end``, so a binary search would be unsound.
        """
        limit = min(total_len, start + self.absolute_max)
        if limit <= end:
            return end

        # csum[k] = anomalies in [start, start + k)  ->  ratio at length k is csum[k] / k
        csum = np.concatenate(([0], np.cumsum(labels[start:limit], dtype=np.int64)))
        lengths = np.arange(csum.size)
        ratios = np.divide(csum, lengths, out=np.full(csum.size, np.inf), where=lengths > 0)

        lo = (end - start) + 1  # first candidate length strictly beyond the current end
        if lo >= csum.size:
            return end

        for threshold in (self.legacy_target_anomaly_ratio, self.legacy_acceptable_anomaly_ratio):
            hits = np.flatnonzero(ratios[lo:] <= threshold)
            if hits.size:
                return start + lo + int(hits[0])
        return end

    # ----------------------------------------------------------------------------------
    # Sample construction
    # ----------------------------------------------------------------------------------

    def _finalize_sample(
        self,
        series_df: pd.DataFrame,
        group: str,
        dataset: str,
        orig_id: str,
        sample_id: str,
        start: int,
        end: int,
        is_split: bool,
        orig_len: int,
        g_stats: Dict,
        period: Optional[int],
        notes: str = "",
        ratio_cap: Optional[float] = None,
    ) -> Optional[SampleRecord]:
        """Validate length and representativeness, then build the record.

        The only place a :class:`SampleRecord` is created, so length bounds and label/ratio
        bookkeeping cannot diverge between code paths.
        """
        end = min(end, start + self.absolute_max, orig_len)
        if end - start < self.absolute_min:
            return None

        chunk = series_df.iloc[start:end]
        values = chunk["value"].to_numpy()
        labels = chunk["label"].to_numpy()
        y_i, anom_count, ratio = self._recalculate_label(labels)

        representative = True
        if y_i == 0:
            # Only clean samples are screened: an anomalous chunk is expected to deviate,
            # so the test would reject exactly what we want to keep. A clean-labelled
            # chunk that is wildly unlike its parent series is suspicious of unlabelled
            # anomalies (known TSB-UAD label noise), hence the rejection.
            representative = self.check_representativeness(values, g_stats["mean"], g_stats["std"])
            if not representative and self.legacy:
                # Legacy only: try to rescue the chunk by extending it. Dropped from the
                # current logic - extending y=0 samples but never y=1 samples ties
                # length to the label.
                expanded_end = min(orig_len, start + self.absolute_max, end + min((end - start) // 5, 2000))
                if expanded_end > end:
                    candidate = series_df["value"].to_numpy()[start:expanded_end]
                    if self.check_representativeness(candidate, g_stats["mean"], g_stats["std"]):
                        end = expanded_end
                        chunk = series_df.iloc[start:end]
                        values = chunk["value"].to_numpy()
                        labels = chunk["label"].to_numpy()
                        y_i, anom_count, ratio = self._recalculate_label(labels)
                        representative = True
            if not representative:
                self.rejected_count += 1
                self.rejected_repr += 1
                return None

        if not self.legacy:
            effective_cap = self.max_anomaly_ratio
        else:
            effective_cap = self.legacy_acceptable_anomaly_ratio if ratio_cap is None else ratio_cap
        if ratio > effective_cap:
            self.rejected_count += 1
            self.rejected_budget += 1
            return None

        note_parts = [p for p in (f"period={period}" if period else "", notes) if p]
        return SampleRecord(
            series_id=f"{group}__{dataset}__{orig_id}_{sample_id}",
            time_index=np.arange(end - start, dtype=np.int64),
            value=values,
            label=labels,
            length=end - start,
            num_point_anomalies=anom_count,
            y_i=y_i,
            is_split=is_split,
            original_length=orig_len,
            source_notes=";".join(note_parts) or None,
            period_detected=period,
            is_representative=representative,
            anomaly_ratio=ratio,
            _start_idx=start,
            _end_idx=end,
        )

    def _create_chunk_sample(
        self,
        series_df: pd.DataFrame,
        group: str,
        dataset: str,
        orig_id: str,
        sample_id: str,
        start: int,
        end: int,
        is_split: bool,
        orig_len: int,
        g_stats: Dict,
        period: Optional[int],
    ) -> Optional[SampleRecord]:
        """Apply the local anomaly budget to one chunk, diluting it if necessary (legacy).

        Purely chunk-local; whole-series strategies are decided in :meth:`process_series`.
        Outside the legacy mode dilution is disabled: the boundary must not move in
        response to the labels, so the budget is enforced by rejection inside
        :meth:`_finalize_sample`.
        """
        if self.legacy:
            labels_full = series_df["label"].to_numpy()
            _, anom_count, ratio = self._recalculate_label(labels_full[start:end])

            if ratio > self.legacy_target_anomaly_ratio:
                expanded_end = self._optimize_expansion(labels_full, start, end, orig_len)
                if expanded_end > end:
                    end = expanded_end

        return self._finalize_sample(
            series_df, group, dataset, orig_id, sample_id, start, end,
            is_split, orig_len, g_stats, period,
        )

    def _extract_clean_chunks(
        self,
        series_df: pd.DataFrame,
        group: str,
        dataset: str,
        orig_id: str,
        period: Optional[int],
        g_stats: Dict,
    ) -> List[SampleRecord]:
        """Salvage the anomaly-free stretches of a series that is mostly anomalous.

        Spans are clean by construction: alignment is applied only when it does not pull a
        labelled point into the span.
        """
        labels = series_df["label"].to_numpy()
        n = labels.size
        orig_len = n
        samples: List[SampleRecord] = []
        pos = 0

        while pos < n:
            if labels[pos] == 1:
                pos += 1
                continue

            limit = min(n, pos + self.absolute_max)
            run = np.flatnonzero(labels[pos:limit] == 1)
            end = pos + int(run[0]) if run.size else limit

            aligned = self.align_boundary(end, period, n, is_start=False)
            if aligned > end and aligned <= limit and not labels[end:aligned].any():
                end = aligned

            if end - pos >= self.absolute_min:
                record = self._finalize_sample(
                    series_df, group, dataset, orig_id, f"clean{len(samples)}",
                    pos, end, True, orig_len, g_stats, period, notes="clean_salvage",
                )
                if record is not None:
                    samples.append(record)
                    pos = record._end_idx
                    continue
            pos = max(end, pos + 1)

        return samples

    def _merge_tail(
        self, last: SampleRecord, values: np.ndarray, labels: np.ndarray, start: int, end: int
    ) -> Optional[SampleRecord]:
        """Append a too-short trailing span to the previous sample.

        Returns ``None`` if the merge would break the length or anomaly budget; the caller
        then drops the tail rather than emitting a sample that violates the invariants.
        """
        merged_value = np.concatenate([last.value, values[start:end]])
        if merged_value.size > self.absolute_max:
            return None

        merged_label = np.concatenate([last.label, labels[start:end]])
        y_i, anom_count, ratio = self._recalculate_label(merged_label)
        if ratio > self.legacy_acceptable_anomaly_ratio:
            return None

        return SampleRecord(
            series_id=last.series_id,
            time_index=np.arange(merged_value.size, dtype=np.int64),
            value=merged_value,
            label=merged_label,
            length=merged_value.size,
            num_point_anomalies=anom_count,
            y_i=y_i,
            is_split=True,
            original_length=last.original_length,
            source_notes=";".join(p for p in (last.source_notes, "tail_merged") if p),
            period_detected=last.period_detected,
            is_representative=last.is_representative,
            anomaly_ratio=ratio,
            _start_idx=last._start_idx,
            _end_idx=end,
        )

    # ----------------------------------------------------------------------------------
    # Series-level orchestration
    # ----------------------------------------------------------------------------------

    def process_series(
        self, series_df: pd.DataFrame, group: str, dataset: str, orig_id: str
    ) -> List[SampleRecord]:
        """Turn one raw series into zero or more samples."""
        values = series_df["value"].to_numpy()
        labels = series_df["label"].to_numpy()
        length = values.size
        if length == 0:
            return []

        if not self.legacy:
            g_stats = robust_reference(values)
        else:
            g_stats = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        period = self.detect_period(values) if length >= MIN_LENGTH_FOR_PERIOD else None

        if not self.legacy:
            # Uniform path for every series: short ones stay whole at their natural
            # length, long ones go on a fixed grid. No label-driven strategies exist, so
            # there is nothing to triage - a degenerate series simply loses its dirty
            # chunks to rejection and keeps its clean ones.
            if length <= self.split_threshold:
                record = self._finalize_sample(
                    series_df, group, dataset, orig_id, "full", 0, length,
                    False, length, g_stats, period,
                )
                return [record] if record else []
            return self._chunk_series_v2(series_df, group, dataset, orig_id, period, g_stats)

        global_ratio = float(np.sum(labels)) / length

        # --- Whole-series triage, decided exactly once ---------------------------------
        # Chunking cannot rescue a series whose global ratio already exceeds the budget.
        # Deciding this up front, not per chunk, keeps series_id unique and coverage
        # disjoint.
        if global_ratio > self.legacy_max_anomaly_ratio:
            self.degenerate_count += 1
            return self._extract_clean_chunks(series_df, group, dataset, orig_id, period, g_stats)

        if global_ratio > self.legacy_acceptable_anomaly_ratio and length <= self.absolute_max:
            record = self._finalize_sample(
                series_df, group, dataset, orig_id, "full", 0, length,
                False, length, g_stats, period, notes="whole_series",
                ratio_cap=self.legacy_max_anomaly_ratio,
            )
            return [record] if record else []

        # --- Short series: a single sample --------------------------------------------
        if length <= self.split_threshold:
            record = self._create_chunk_sample(
                series_df, group, dataset, orig_id, "full", 0, length,
                False, length, g_stats, period,
            )
            return [record] if record else []

        # --- Long series: disjoint chunks ---------------------------------------------
        return self._chunk_series(series_df, group, dataset, orig_id, period, g_stats)

    def _chunk_series_v2(
        self,
        series_df: pd.DataFrame,
        group: str,
        dataset: str,
        orig_id: str,
        period: Optional[int],
        g_stats: Dict,
    ) -> List[SampleRecord]:
        """Split a long series on a fixed, label-independent grid.

        The chunk size is computed once per series from the values alone and already
        rounded to a whole number of periods, so every boundary falls on a period
        multiple by construction - no per-chunk alignment, no expansion, no merging.
        Within one series every chunk has the same length (the last may be shorter, a
        function of the series length only). Rejection never moves the grid: a dropped
        slot stays dropped, and its index stays burned, so `chunk{N}` names the position
        in the original series rather than the emission order.
        """
        length = len(series_df)
        chunk_size = self.compute_optimal_chunk_size(period)

        samples: List[SampleRecord] = []
        pos = 0
        chunk_idx = 0
        while length - pos >= self.absolute_min:
            end = min(pos + chunk_size, length)
            record = self._finalize_sample(
                series_df, group, dataset, orig_id, f"chunk{chunk_idx}", pos, end,
                True, length, g_stats, period,
            )
            if record is not None:
                samples.append(record)
            pos = end
            chunk_idx += 1
        return samples

    def _chunk_series(
        self,
        series_df: pd.DataFrame,
        group: str,
        dataset: str,
        orig_id: str,
        period: Optional[int],
        g_stats: Dict,
    ) -> List[SampleRecord]:
        """Split a long series into non-overlapping, budget-respecting chunks (legacy)."""
        values = series_df["value"].to_numpy()
        labels = series_df["label"].to_numpy()
        length = values.size
        chunk_size = self.compute_optimal_chunk_size(period)

        samples: List[SampleRecord] = []
        pos = 0
        chunk_idx = 0

        while pos < length:
            start = pos
            end = min(pos + chunk_size, length)
            end = self.align_boundary(end, period, length, is_start=False)
            if np.any(labels[start:end] == 1):
                start, end = self.expand_anomaly_cluster(labels, start, end, period=period)

            # Too short to stand alone: merge into the previous sample, else drop.
            if end - start < self.absolute_min:
                if samples:
                    merged = self._merge_tail(samples[-1], values, labels, start, end)
                    if merged is not None:
                        samples[-1] = merged
                    else:
                        self.rejected_count += 1
                pos = max(end, start + 1)
                continue

            record = self._create_chunk_sample(
                series_df, group, dataset, orig_id, f"chunk{chunk_idx}", start, end,
                True, length, g_stats, period,
            )
            if record is not None:
                samples.append(record)
                chunk_idx += 1
                # Resume from the expanded boundary - this is what makes chunks disjoint.
                pos = max(record._end_idx, start + 1)
            else:
                pos = max(end, start + 1)

        return samples

    # ----------------------------------------------------------------------------------
    # Group-level orchestration
    # ----------------------------------------------------------------------------------

    @staticmethod
    def _load_series(path: Path) -> Optional[pd.DataFrame]:
        """Read one raw file and normalise it to a ``(value, label)`` frame."""
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if df.empty:
            return None
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "data" in df.columns and "value" not in df.columns:
            df = df.rename(columns={"data": "value"})
        if not {"value", "label"}.issubset(df.columns):
            return None

        df = df[["value", "label"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
        df["label"] = (pd.to_numeric(df["label"], errors="coerce").fillna(0) > 0).astype("int8")
        # TSB-UAD rule: forward-fill gaps, drop any remaining at the head.
        df["value"] = df["value"].ffill()
        df = df[df["value"].notna()].reset_index(drop=True)
        return df if len(df) else None

    def process_group(self, raw_dir: Path, group: str, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process every dataset directory of a group and persist the pool."""
        self.rejected_count = 0
        self.rejected_budget = 0
        self.rejected_repr = 0
        self.degenerate_count = 0
        # (dataset, recording, record): identity kept explicit so the composition stage
        # never has to re-parse it out of series_id.
        entries: List[Tuple[str, str, SampleRecord]] = []

        if not raw_dir.exists():
            logger.error("Raw directory not found: %s", raw_dir)
            return pd.DataFrame(), pd.DataFrame()

        dataset_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
        logger.info("=== Starting %s: found %d datasets ===", group, len(dataset_dirs))

        for ds_idx, ds_dir in enumerate(dataset_dirs, 1):
            ds_name = ds_dir.name
            ds_start = time.time()
            files = sorted(list(ds_dir.glob("*.csv")) + list(ds_dir.glob("*.parquet")))
            if not files:
                logger.warning("  [%d/%d] %s: no .csv/.parquet files", ds_idx, len(dataset_dirs), ds_name)
                continue
            logger.info("  [%d/%d] %s: %d files", ds_idx, len(dataset_dirs), ds_name, len(files))

            produced = 0
            for file_idx, fpath in enumerate(files, 1):
                try:
                    df = self._load_series(fpath)
                    if df is None:
                        continue
                    for sample in self.process_series(df, group, ds_name, fpath.stem):
                        entries.append((ds_name, fpath.stem, sample))
                        produced += 1
                except Exception as exc:  # one broken file must not abort the pool
                    logger.error("    Error %s: %s", fpath.name, exc)
                if file_idx % 200 == 0:
                    logger.info("    ... %d/%d files", file_idx, len(files))

            logger.info("  [ok] %s: %d samples in %.1fs", ds_name, produced, time.time() - ds_start)

        if not entries:
            logger.warning("No valid samples for %s", group)
            return pd.DataFrame(), pd.DataFrame()

        output_dir.mkdir(parents=True, exist_ok=True)
        composition: Dict[str, object] = {}
        full_counts: Dict[str, object] = {}
        split_map: Optional[Dict[Tuple[str, str], str]] = None
        if not self.legacy:
            full_entries = entries
            entries, composition = self._compose_pool(entries)
            # The quota targets the pool people evaluate on; the full pools inherit the
            # same machine->side mapping so a series never changes sides between them.
            split_map = self._split_mapping(entries, group)
            if WRITE_FULL_POOLS:
                # The uncomposed variant: every valid sample, no size/rate/share shaping.
                # The grouped split is still assigned - it marks independence, not size.
                full_main, full_meta = self._build_frames(full_entries, group, split_map)
                validate_pool(full_main, full_meta, self)
                FULL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                full_main.to_parquet(FULL_OUTPUT_DIR / f"{group}.parquet", index=False)
                full_meta.to_parquet(FULL_OUTPUT_DIR / f"{group}_metadata.parquet", index=False)
                full_counts = {
                    "full_samples": int(len(full_meta)),
                    "full_positive_rate": round(float(full_meta["y_i"].mean()), 4),
                }
                logger.info(
                    "Full pool: %d samples (%.1f%% positive) -> full/%s.parquet",
                    len(full_meta), 100 * full_meta["y_i"].mean(), group,
                )

        main_df, meta_df = self._build_frames(entries, group, split_map)
        validate_pool(main_df, meta_df, self)

        main_df.to_parquet(output_dir / f"{group}.parquet", index=False)
        meta_df.to_parquet(output_dir / f"{group}_metadata.parquet", index=False)
        if not self.legacy:
            manifest = {
                "max_anomaly_ratio": self.max_anomaly_ratio,
                "target_pool_size": TARGET_POOL_SIZE,
                "target_positive_rate": TARGET_POSITIVE_RATE,
                "max_recording_share": MAX_RECORDING_SHARE,
                "max_samples_per_recording": MAX_SAMPLES_PER_RECORDING,
                "composition_seed": COMPOSITION_SEED,
                "dev_fraction": DEV_FRACTION,
                **composition,
                **full_counts,
                "final_samples": int(len(meta_df)),
                "final_positive_rate": round(float(meta_df["y_i"].mean()), 4),
            }
            (output_dir / f"{group}_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

        self._log_summary(group, meta_df, output_dir)
        return main_df, meta_df

    def _split_mapping(
        self, entries: List[Tuple[str, str, SampleRecord]], group: str
    ) -> Dict[Tuple[str, str], str]:
        """Assign whole machines to dev/test - a stratified grouped split.

        The unit stays the machine-level identity - columns of one machine, trace or ECG
        record share regimes and anomaly windows, so they must land on one side. Machines
        are walked in stable-hash order and placed in dev whenever that brings the dev
        side closer to BOTH quotas at once: DEV_FRACTION of the samples and DEV_FRACTION
        of the positives (the greedy analogue of a stratified grouped split). Without the
        second term the sides drift apart in class balance and a threshold tuned on dev
        stops transferring to test. Deterministic; overshoot is bounded by one machine,
        so the realised shares track the targets as closely as atomic machines allow.
        """
        counts: Dict[Tuple[str, str], np.ndarray] = {}
        for ds, rec_id, rec in entries:
            key = (ds, recording_group(rec_id))
            counts.setdefault(key, np.zeros(2))[:] += (1, rec.y_i)
        order = sorted(counts, key=lambda k: stable_rank(COMPOSITION_SEED, "split", group, *k))
        target = DEV_FRACTION * sum(counts.values())  # (samples, positives)
        norm = np.maximum(target, 1.0)

        def distance(state: np.ndarray) -> float:
            return float(np.sum(np.abs(state - target) / norm))

        mapping: Dict[Tuple[str, str], str] = {}
        dev = np.zeros(2)
        for key in order:
            take = distance(dev + counts[key]) < distance(dev)
            mapping[key] = "dev" if take else "test"
            if take:
                dev += counts[key]

        # The greedy is myopic, and with lumpy machines (in R2 every capped recording is
        # ~5% of the pool) it can strand the class balance in a local miss. Polish with
        # deterministic local search: single flips move both quotas at once, dev<->test
        # swaps adjust the positive balance while barely touching the sample count.
        # Monotone improvement over a finite state space, so the loop terminates.
        improved = True
        while improved:
            improved = False
            for key in order:
                delta = counts[key] if mapping[key] == "test" else -counts[key]
                if distance(dev + delta) < distance(dev):
                    mapping[key] = "dev" if mapping[key] == "test" else "test"
                    dev += delta
                    improved = True
            for a in order:
                if mapping[a] != "dev":
                    continue
                for b in order:
                    if mapping[b] != "test":
                        continue
                    delta = counts[b] - counts[a]
                    if distance(dev + delta) < distance(dev):
                        mapping[a], mapping[b] = "test", "dev"
                        dev += delta
                        improved = True
                        break

        if len(counts) > 1 and len(set(mapping.values())) == 1:
            # a degenerate quota must not empty either side
            flip = order[0]
            mapping[flip] = "dev" if mapping[flip] == "test" else "test"
        return mapping

    def _build_frames(
        self,
        entries: List[Tuple[str, str, SampleRecord]],
        group: str,
        split_map: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Assemble the data and metadata frames for a set of samples.

        The grouped dev/test split lands here, from the machine->side mapping of
        :meth:`_split_mapping`; a machine absent from that mapping (present only in the
        full pool because composition dropped it entirely) falls back to the plain hash
        rule, so it still gets a stable side. Correlated series can never straddle the
        evaluation boundary - which is also asserted, not assumed.
        """
        main_df = pd.concat([rec.to_dataframe() for _, _, rec in entries], ignore_index=True)
        meta_df = pd.DataFrame([rec.to_metadata_row() for _, _, rec in entries])
        main_df["time_index"] = main_df["time_index"].astype(np.int64)
        main_df["value"] = main_df["value"].astype(np.float64)
        main_df["label"] = main_df["label"].astype(np.int8)

        if not self.legacy:
            keys = [(ds, recording_group(rec_id)) for ds, rec_id, _ in entries]
            hash_side = lambda key: (
                "dev" if stable_rank(COMPOSITION_SEED, "split", group, *key) < DEV_FRACTION
                else "test"
            )
            meta_df["split"] = [
                (split_map or {}).get(key) or hash_side(key) for key in keys
            ]
            machine = pd.Series([f"{ds}__{m}" for ds, m in keys])
            per_split = {s: set(machine[meta_df["split"] == s]) for s in ("dev", "test")}
            overlap = per_split["dev"] & per_split["test"]
            if overlap:
                raise AssertionError(f"machines straddle dev/test: {sorted(overlap)[:5]}")
        return main_df, meta_df

    # ----------------------------------------------------------------------------------
    # Pool composition
    # ----------------------------------------------------------------------------------

    @staticmethod
    def _waterfill(avail: Dict[Tuple[str, str], int], target: int) -> Dict[Tuple[str, str], int]:
        """Most balanced allocation of ``target`` slots across recordings.

        One ceiling is shared by all recordings and raised until the target is met:
        recordings with fewer samples than the ceiling contribute everything they have
        ("this recording ran out"), larger ones contribute exactly the ceiling. The
        remainder below the final ceiling is distributed one slot each in stable hash
        order, so the result is deterministic and independent of processing order.
        """
        total = sum(avail.values())
        if target <= 0:
            return {key: 0 for key in avail}
        if target >= total:
            return dict(avail)

        lo, hi = 1, max(avail.values())
        while lo < hi:  # smallest ceiling whose capacity reaches the target
            mid = (lo + hi) // 2
            if sum(min(v, mid) for v in avail.values()) >= target:
                hi = mid
            else:
                lo = mid + 1
        ceiling = lo

        counts = {key: min(v, ceiling - 1) for key, v in avail.items()}
        remainder = target - sum(counts.values())
        at_ceiling = sorted(
            (key for key, v in avail.items() if v >= ceiling),
            key=lambda key: stable_rank(COMPOSITION_SEED, "fill", *key),
        )
        for key in at_ceiling[:remainder]:
            counts[key] += 1
        return counts

    def _compose_pool(
        self, entries: List[Tuple[str, str, SampleRecord]]
    ) -> Tuple[List[Tuple[str, str, SampleRecord]], Dict[str, object]]:
        """Assemble the benchmark pool at the largest size every constraint allows.

        The size aims at ``TARGET_POOL_SIZE`` and is lowered only when a constraint
        demands it: the positive class must supply ``TARGET_POSITIVE_RATE`` of the pool,
        the negative class the rest, and under ``MAX_RECORDING_SHARE`` no recording may
        exceed that share of the final pool. Within the chosen size, samples are spread
        across recordings by :meth:`_waterfill` - positives first (the scarce class),
        then negatives into each recording's remaining allowance.

        Nothing here looks at values or lengths - only ``y_i`` and identity - so the
        composition cannot introduce a length- or content-based leak. Which samples of a
        recording take its slots is fixed by a hash of their own ids (spread evenly over
        the recording, not taken from its head); only the water level moves when the
        pool around them changes, which any exact size target requires.
        """
        stats: Dict[str, object] = {"emitted_samples": len(entries)}

        by_rec: Dict[Tuple[str, str], Dict[int, List[Tuple[str, str, SampleRecord]]]] = {}
        for entry in entries:
            by_rec.setdefault((entry[0], entry[1]), {0: [], 1: []})[entry[2].y_i].append(entry)
        for classes in by_rec.values():
            for lst in classes.values():
                lst.sort(key=lambda e: stable_rank(COMPOSITION_SEED, "pick", e[2].series_id))
        stats["recordings"] = len(by_rec)

        # Optional legacy hard cap, kept as a fallback control.
        if MAX_SAMPLES_PER_RECORDING is not None:
            for classes in by_rec.values():
                merged = sorted(
                    classes[0] + classes[1],
                    key=lambda e: stable_rank(COMPOSITION_SEED, "pick", e[2].series_id),
                )[:MAX_SAMPLES_PER_RECORDING]
                kept_ids = {e[2].series_id for e in merged}
                classes[0] = [e for e in classes[0] if e[2].series_id in kept_ids]
                classes[1] = [e for e in classes[1] if e[2].series_id in kept_ids]

        n_pos = sum(len(c[1]) for c in by_rec.values())
        n_neg = sum(len(c[0]) for c in by_rec.values())
        rate = TARGET_POSITIVE_RATE
        big = 10**9

        # With R recordings some recording always holds at least 1/R of the pool, so a
        # share cap below that floor is unsatisfiable at *any* size. Detect it here and
        # relax to the floor: silently chasing an impossible cap would shrink the pool
        # toward R samples instead of reporting the conflict.
        share = MAX_RECORDING_SHARE
        if share is not None:
            share_floor = 1.0 / len(by_rec)
            if share < share_floor:
                logger.warning(
                    "MAX_RECORDING_SHARE=%.4f is below the 1/%d=%.4f floor implied by the "
                    "number of recordings; relaxing to the floor (even split)",
                    share, len(by_rec), share_floor,
                )
                share = share_floor
            stats["effective_recording_share"] = round(share, 6)

        def ceiling_for(n: int) -> int:
            # Rounded UP: samples are indivisible, and truncating here would make
            # feasibility non-monotone in the size (with share = 1/R the capacity
            # R*floor(n/R) only reaches n when n is a multiple of R), which a binary
            # search over sizes cannot handle. Rounding up costs at most one sample per
            # recording and keeps capacity >= n whenever share >= 1/R.
            # The epsilon absorbs float error: 0.05 * 2900 evaluates to 145.00000000000003,
            # and a bare ceil would inflate the ceiling by one for no reason.
            return max(1, math.ceil(share * n - 1e-9)) if share is not None else big

        def feasible(n: int) -> bool:
            """Whether a pool of exactly ``n`` samples satisfies every active control."""
            if n <= 0:
                return True
            ceiling = ceiling_for(n)
            if rate is None:
                return sum(min(len(c[0]) + len(c[1]), ceiling) for c in by_rec.values()) >= n
            need_pos = int(round(n * rate))
            pos_counts = self._waterfill(
                {key: min(len(c[1]), ceiling) for key, c in by_rec.items()}, need_pos
            )
            if sum(pos_counts.values()) < need_pos:
                return False
            neg_cap = sum(
                min(len(c[0]), ceiling - pos_counts.get(key, 0)) for key, c in by_rec.items()
            )
            return neg_cap >= n - need_pos

        # Largest size every control allows. Binary search rather than a fixpoint
        # iteration: feasibility is monotone in the size once the share floor is
        # respected, and a search cannot spiral.
        upper = n_pos + n_neg
        if rate is not None:
            upper = min(upper, int(n_pos / rate) if rate > 0 else upper,
                        int(n_neg / (1.0 - rate)) if rate < 1 else upper)
        if TARGET_POOL_SIZE is not None:
            upper = min(upper, TARGET_POOL_SIZE)

        lo, hi = 0, max(0, upper)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if feasible(mid):
                lo = mid
            else:
                hi = mid - 1
        size = lo
        ceiling = ceiling_for(size) if share is not None else None
        if ceiling is not None:
            stats["recording_ceiling"] = ceiling
        if TARGET_POOL_SIZE is not None and size < TARGET_POOL_SIZE:
            logger.warning(
                "pool built at %d samples, below TARGET_POOL_SIZE=%d: the positive-rate "
                "and per-recording-share controls do not allow more",
                size, TARGET_POOL_SIZE,
            )

        if rate is None:
            merged_by_rec = {
                key: sorted(c[0] + c[1], key=lambda e: stable_rank(COMPOSITION_SEED, "pick", e[2].series_id))
                for key, c in by_rec.items()
            }
            counts = self._waterfill(
                {key: min(len(lst), ceiling or big) for key, lst in merged_by_rec.items()}, size
            )
            kept = [e for key, lst in merged_by_rec.items() for e in lst[: counts[key]]]
        else:
            need_pos = int(round(size * rate))
            pos_counts = self._waterfill(
                {key: min(len(c[1]), ceiling or big) for key, c in by_rec.items()}, need_pos
            )
            neg_counts = self._waterfill(
                {
                    key: min(len(c[0]), (ceiling or big) - pos_counts.get(key, 0))
                    for key, c in by_rec.items()
                },
                size - sum(pos_counts.values()),
            )
            kept = []
            for key, c in by_rec.items():
                kept.extend(c[1][: pos_counts.get(key, 0)])
                kept.extend(c[0][: neg_counts.get(key, 0)])

        # Restore the deterministic global order (dataset, recording, chunk position).
        kept.sort(key=lambda e: (e[0], e[1], e[2]._start_idx))
        stats["target_pool_size"] = TARGET_POOL_SIZE
        stats["feasible_pool_size"] = size
        stats["final_positives"] = sum(e[2].y_i for e in kept)
        if len(kept) < size:
            logger.warning("composition fell short of the feasible size: %d < %d", len(kept), size)
        return kept, stats

    def _log_summary(self, group: str, meta_df: pd.DataFrame, output_dir: Path) -> None:
        logger.info("=== %s COMPLETE%s ===", group, " (legacy)" if self.legacy else "")
        logger.info("Total samples: %d", len(meta_df))
        logger.info(
            "Length min/mean/max/std: %d / %.1f / %d / %.1f",
            meta_df["length"].min(), meta_df["length"].mean(),
            meta_df["length"].max(), meta_df["length"].std(),
        )
        logger.info(
            "Anomalous (y_i=1): %d (%.1f%%)", meta_df["y_i"].sum(), 100 * meta_df["y_i"].mean()
        )
        logger.info(
            "Anomaly ratio mean/max: %.4f / %.4f",
            meta_df["anomaly_ratio"].mean(), meta_df["anomaly_ratio"].max(),
        )
        logger.info("Period detected: %d (%.1f%%)",
                    meta_df["period_detected"].notna().sum(),
                    100 * meta_df["period_detected"].notna().mean())
        if "split" in meta_df.columns:
            dev = meta_df[meta_df["split"] == "dev"]
            test = meta_df[meta_df["split"] == "test"]
            logger.info(
                "Split: dev %d (pos %.1f%%) / test %d (pos %.1f%%)",
                len(dev), 100 * dev["y_i"].mean() if len(dev) else 0.0,
                len(test), 100 * test["y_i"].mean() if len(test) else 0.0,
            )
        logger.info(
            "Rejected samples: %d (anomaly budget %d, representativeness %d)",
            self.rejected_count, self.rejected_budget, self.rejected_repr,
        )
        logger.info("Degenerate series (clean salvage): %d", self.degenerate_count)
        logger.info("Saved to %s/%s.parquet", output_dir, group)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def validate_pool(data: pd.DataFrame, meta: pd.DataFrame, sampler: TimeSeriesSampler) -> None:
    """Assert the invariants promised in the module docstring, before anything is written."""
    if meta["series_id"].duplicated().any():
        dupes = meta.loc[meta["series_id"].duplicated(), "series_id"].tolist()[:5]
        raise AssertionError(f"duplicate series_id, e.g. {dupes}")
    if not np.isfinite(data["value"].to_numpy()).all():
        raise AssertionError("value contains NaN or inf")
    if not data["label"].isin((0, 1)).all():
        raise AssertionError("label must be binary")

    too_short = meta[meta["length"] < sampler.absolute_min]
    if len(too_short):
        raise AssertionError(f"{len(too_short)} samples shorter than absolute_min")
    too_long = meta[meta["length"] > sampler.absolute_max]
    if len(too_long):
        raise AssertionError(f"{len(too_long)} samples longer than absolute_max")

    if not sampler.legacy:
        # One uniform rejection cap for every sample.
        cap = np.full(len(meta), sampler.max_anomaly_ratio)
    else:
        # Legacy: chunk samples are held to the acceptable bound; whole-series to the max.
        whole = meta["source_notes"].fillna("").str.contains("whole_series")
        cap = np.where(whole, sampler.legacy_max_anomaly_ratio, sampler.legacy_acceptable_anomaly_ratio)
    over_budget = meta[meta["anomaly_ratio"] > cap + 1e-9]
    if len(over_budget):
        worst = over_budget["anomaly_ratio"].max()
        raise AssertionError(
            f"{len(over_budget)} samples exceed their anomaly-ratio budget (worst {worst:.4f})"
        )

    observed = data.groupby("series_id", sort=False).agg(
        obs_length=("time_index", "size"),
        obs_anomalies=("label", "sum"),
        last_index=("time_index", "max"),
    )
    joined = meta.set_index("series_id").join(observed, how="left")
    if joined["obs_length"].isna().any():
        raise AssertionError("metadata references a series_id absent from the data frame")
    if not (joined["obs_length"] == joined["length"]).all():
        raise AssertionError("metadata length disagrees with the number of rows")
    if not (joined["obs_anomalies"] == joined["num_point_anomalies"]).all():
        raise AssertionError("num_point_anomalies disagrees with the labels")
    if not (joined["y_i"] == (joined["obs_anomalies"] > 0).astype(int)).all():
        raise AssertionError("y_i disagrees with the point-wise labels")
    if not (joined["last_index"] == joined["length"] - 1).all():
        raise AssertionError("time_index must be contiguous and start at 0 for every series")
    if "split" in meta.columns and not meta["split"].isin(("dev", "test")).all():
        raise AssertionError("split must be 'dev' or 'test' for every sample")


def main() -> None:
    """Build every pool listed in GROUPS_TO_BUILD."""
    sampler = TimeSeriesSampler()
    for group, subdir in GROUPS_TO_BUILD.items():
        logger.info("=== Processing %s ===", group)
        sampler.process_group(RAW_DATA_DIR / subdir, group, OUTPUT_DIR)


if __name__ == "__main__":
    main()
