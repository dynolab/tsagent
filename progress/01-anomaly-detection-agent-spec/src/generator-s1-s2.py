"""Synthetic univariate time-series generator for the S1/S2 anomaly-detection pools.

Every tunable knob lives in the CONFIGURATION block below.

Guarantees
----------
Reproducibility
    One independent child RNG per series, so a series depends only on
    ``(SEED, group, series_index)`` - not on pool size, pool order or parallelism.
    Each run writes a ``*_manifest.json`` with the exact settings used.

Severity scale
    Anomalies are sized against the visible geometry of the particular series
    (baseline, band ``sigma_vis``, reach of its own extremes): segment families
    displace by ``severity * sigma_vis`` with two-sided floors/caps against the
    masking the series itself produces, and ``point`` spikes land a
    severity-controlled margin BEYOND the series' own reach.

Labelling
    Perturbations are tapered to exactly zero at both segment boundaries, so the perturbed
    support equals the labelled support and no unlabelled discontinuity is left behind.

Anomaly taxonomy
    ``point`` isolated outliers | ``group`` alien shapelet | ``level_shift`` mean step |
    ``variance`` dispersion inflation | ``trend`` transient drift | ``seasonality`` period
    warp - six mechanically distinct families.
"""

from __future__ import annotations

import json
import logging
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import lfilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================================
# CONFIGURATION - every knob for synthetic generation lives in this block
# ======================================================================================

# Destination for S1/S2, their metadata and the run manifests. Resolved from this file,
# so the script behaves the same whatever directory it is launched from.
_HERE = Path(__file__).resolve()
OUTPUT_DIR = _HERE.parents[3] / "data" / _HERE.parents[1].name / "synthetic"

# Which pools to build on this run, e.g. ("S1",) to rebuild only the stationary pool.
GROUPS_TO_BUILD: tuple[str, ...] = ("S1", "S2")

SEED = 42                               # root of all randomness; fixes the pools exactly
NUM_SERIES_PER_POOL = 2000              # series per pool
LENGTH_RANGE = (1000, 4000)             # per-series length, drawn uniformly
ANOMALY_RATE = 0.25                     # share of series that receive an anomaly: ~500
                                        # dirty series keep ~100 per family for
                                        # per-family analysis, while ~1500 clean series
                                        # give detectors a serious false-positive surface
# Share of anomalous points inside a dirty series. Real pools span two orders of
# magnitude (R1 median ~0.011, R2 median ~0.12), so the fraction is drawn log-uniformly:
# every order of magnitude gets equal coverage, and quality-vs-density curves get support
# across the whole axis instead of a narrow band.
ANOMALY_FRACTION_RANGE = (0.01, 0.20)

# --- Realism knobs ---
P_HEAVY_TAIL = 0.25             # share of series with Student-t innovations: real sensor
                                # noise has heavy tails, and Gaussian-only negatives make
                                # every detector look better than it is
HEAVY_TAIL_DF = (4.0, 10.0)     # t-distribution degrees of freedom (lower = heavier)
HEAVY_TAIL_CLIP = 3.5           # heavy-tail innovations are REDRAWN beyond this many
                                # sigma: frequent 2.5-3.5 sigma shoulders stay (hard
                                # negatives), but a clean series never manufactures a
                                # lone spike bigger than the labelled anomalies.
NOISE_PHI_RANGE = (0.0, 0.7)    # AR(1) coefficient of the noise under trend/seasonal
                                # bases; 0 recovers white noise as a special case
P_SECOND_SEASON = 0.35          # nested longer seasonality (daily-inside-weekly pattern)
P_AMP_MODULATION = 0.40         # slow drift of the seasonal amplitude over the series
BASELINE_OFFSET = (-50.0, 50.0) # per-series constant level; catches detectors that
                                # silently assume zero-centred data
P_TWO_EVENTS = 0.25             # chance a dirty series carries two disjoint events
EVENT_MARGIN = 25               # minimum clean gap between two events, in samples
DEV_FRACTION = 0.30             # share of series hashed into the "dev" split
SEVERITY_RANGE = (2.6, 3.6)     # anomaly size dial, in units of sigma_vis:
                                # level_shift/trend displace by severity*sigma_vis,
                                # group/seasonality span it peak-to-peak, variance maps
                                # it to a dispersion ratio (1 + severity/2), point to a
                                # margin beyond the series' OWN extreme reach.
# Per-family severity ranges. Uncomment a line to give that family its own range; a
# family left commented draws from SEVERITY_RANGE - one range for all by default.
SEVERITY_RANGE_OVERRIDES: dict[str, tuple[float, float]] = {
    "point":       (2.6, 3.1),  # margin past the series' own extremes (_inject_point)
    "group":       (2.8, 3.6),  # peak-to-peak span of the alien waveform, in sigma_vis
    # "level_shift": (2.6, 3.6),  # step height, in sigma_vis
    # "variance":    (2.6, 3.6),  # dispersion ratio = 1 + severity/2
    # "trend":       (2.6, 3.6),  # peak drift, in sigma_vis
    # "seasonality": (2.6, 3.6),  # peak-to-peak of the seasonal discrepancy, in sigma_vis
    # "flatline":    (2.6, 3.6),  # unused by the injector; listed for completeness
}
# Per-family anomalous-point-fraction ranges (drawn log-uniformly, like the global
# range). Uncomment a line to pin a family; left commented, "point" falls back to
# POINT_FRACTION_RANGE and every other family to ANOMALY_FRACTION_RANGE.
ANOMALY_FRACTION_OVERRIDES: dict[str, tuple[float, float]] = {
    # "group":       (0.01, 0.20),
    # "level_shift": (0.01, 0.20),
    # "variance":    (0.01, 0.20),
    # "trend":       (0.01, 0.20),
    # "seasonality": (0.01, 0.20),
    # "flatline":    (0.01, 0.20),
}
POINT_FRACTION_RANGE = (0.004, 0.025)

# --- Detectability certification ---
# Sizing is two-sided: proportional to the series' own band, floored against the
# masking the series itself produces (reach, baseline wander, rolling-scale swing) and
# capped when the drawn size would dwarf it. Certification guards the statistical
# side: every event must reach a whitened matched-filter z of SEGMENT_SNR_FLOOR
# against the series' known AR polynomial (for point, aggregated over the whole spike
# pattern), and sample-size-hungry families get duration floors. The drawn severity
# stays the difficulty axis; the applied size and achieved z are recorded per series.
CERTIFY_DETECTABILITY = True    # False restores uncertified magnitudes and durations
SEGMENT_SNR_FLOOR = 5.0         # min whitened matched-filter z for any injected event
MIN_EVENT_POINTS = {            # per-family duration floors (points per event)
    "group": 128,                # a couple of alien cycles must be legible
    "level_shift": 24,          # a mean step needs points on the shifted side
    "variance": 64,             # a 2-3x dispersion change is invisible on 10 points
    "trend": 48,                # a slow ramp needs room to accumulate
    "seasonality": 48,          # plus the full-cycle-discrepancy rule in the injector
    "flatline": 24,             # the taper must leave a fully flat core
}

# On periodic bases a level shift or drift shorter than a cycle rides one wave flank
# and reads as the wave's own motion (or as the allowed amplitude modulation), so those
# events must span at least this many full periods.
PERIODIC_MIN_PERIODS = 2

# Names must exist in BASE_PROCESSES / ANOMALY_INJECTORS; validated before generation.
POOL_DEFINITIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "S1": {  # Stationary
        "base_types": ("white_noise", "ar1", "ar2"),
        "allowed_anomaly_types": ("point", "group", "level_shift", "variance", "flatline"),
    },
    "S2": {  # Trend-Seasonal
        "base_types": ("linear_trend", "seasonal_sine", "trend_seasonal"),
        "allowed_anomaly_types": ("trend", "seasonality", "group", "level_shift", "flatline", "point"),
    },
}

# --- Structural constants; change only if the generation model itself changes. ---
MIN_SERIES_LENGTH = 64      # below this a tapered segment anomaly cannot be placed
AR_BURN_IN = 500            # samples dropped from an AR draw to remove the transient
POINT_MIN_SPACING = 5       # minimum gap between point outliers so they stay isolated
POINT_EDGE_MARGIN = 50      # keep spikes away from the series' ends: with neighbours on
                            # one side only, any local baseline collapses onto the spike
                            # and its visible displacement reads as zero

# ======================================================================================


@dataclass(frozen=True)
class PoolConfig:
    """Declarative description of one synthetic pool."""

    group: str
    num_series: int
    base_types: Sequence[str]
    allowed_anomaly_types: Sequence[str]
    anomaly_rate: float
    length_range: tuple[int, int]
    anomaly_fraction_range: tuple[float, float]
    severity_range: tuple[float, float]
    seed: int

    def validate(self) -> None:
        """Fail fast on a configuration that cannot produce a valid pool."""
        if self.num_series <= 0:
            raise ValueError(f"num_series must be positive, got {self.num_series}")
        if not 0.0 <= self.anomaly_rate <= 1.0:
            raise ValueError(f"anomaly_rate must lie in [0, 1], got {self.anomaly_rate}")

        lo, hi = self.length_range
        if not MIN_SERIES_LENGTH <= lo <= hi:
            raise ValueError(
                f"length_range must satisfy {MIN_SERIES_LENGTH} <= lo <= hi, got {self.length_range}"
            )

        f_lo, f_hi = self.anomaly_fraction_range
        if not 0.0 < f_lo <= f_hi < 1.0:
            raise ValueError(
                f"anomaly_fraction_range must satisfy 0 < lo <= hi < 1, got {self.anomaly_fraction_range}"
            )

        s_lo, s_hi = self.severity_range
        if not 0.0 < s_lo <= s_hi:
            raise ValueError(f"severity_range must satisfy 0 < lo <= hi, got {self.severity_range}")

        if not self.base_types:
            raise ValueError("base_types must not be empty")
        unknown = sorted(set(self.base_types) - set(BASE_PROCESSES))
        if unknown:
            raise ValueError(f"unknown base_types: {unknown}")

        if self.anomaly_rate > 0 and not self.allowed_anomaly_types:
            raise ValueError("allowed_anomaly_types must not be empty when anomaly_rate > 0")
        unknown = sorted(set(self.allowed_anomaly_types) - set(ANOMALY_INJECTORS))
        if unknown:
            raise ValueError(f"unknown allowed_anomaly_types: {unknown}")

    def to_manifest(self) -> dict:
        """Serialise the configuration for the run manifest."""
        return {
            "group": self.group,
            "num_series": self.num_series,
            "base_types": list(self.base_types),
            "allowed_anomaly_types": list(self.allowed_anomaly_types),
            "anomaly_rate": self.anomaly_rate,
            "length_range": list(self.length_range),
            "anomaly_fraction_range": list(self.anomaly_fraction_range),
            "fraction_sampling": "log_uniform",
            "severity_range": list(self.severity_range),
            "certify_detectability": CERTIFY_DETECTABILITY,
            "segment_snr_floor": SEGMENT_SNR_FLOOR,
            "heavy_tail_clip": HEAVY_TAIL_CLIP,
            "severity_range_overrides": {k: list(v) for k, v in SEVERITY_RANGE_OVERRIDES.items()},
            "anomaly_fraction_overrides": {k: list(v) for k, v in ANOMALY_FRACTION_OVERRIDES.items()},
            "min_event_points": dict(MIN_EVENT_POINTS),
            "point_fraction_range": list(POINT_FRACTION_RANGE),
            "periodic_min_periods": PERIODIC_MIN_PERIODS,
            "seed": self.seed,
        }


def pool_config(group: str) -> PoolConfig:
    """Build the :class:`PoolConfig` for ``group`` from the CONFIGURATION block."""
    if group not in POOL_DEFINITIONS:
        raise KeyError(f"unknown pool {group!r}; known pools: {sorted(POOL_DEFINITIONS)}")
    definition = POOL_DEFINITIONS[group]
    return PoolConfig(
        group=group,
        num_series=NUM_SERIES_PER_POOL,
        base_types=definition["base_types"],
        allowed_anomaly_types=definition["allowed_anomaly_types"],
        anomaly_rate=ANOMALY_RATE,
        length_range=LENGTH_RANGE,
        anomaly_fraction_range=ANOMALY_FRACTION_RANGE,
        severity_range=SEVERITY_RANGE,
        seed=SEED,
    )


@dataclass
class AnomalySegment:
    """A single labelled anomalous span, half-open ``[start, end)``."""

    kind: str
    start: int
    end: int
    severity: float
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "start": int(self.start),
            "end": int(self.end),
            "severity": round(float(self.severity), 4),
            **{k: _jsonable(v) for k, v in self.detail.items()},
        }


@dataclass
class SeriesResult:
    """One generated series together with its provenance."""

    series_id: str
    values: np.ndarray
    labels: np.ndarray
    metadata: dict


def _jsonable(value):
    """Coerce numpy scalars to plain Python types for JSON serialisation."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    return value


# --------------------------------------------------------------------------------------
# Base processes - each returns ``(values, params)``; params is recorded in the metadata.
# --------------------------------------------------------------------------------------

def _innovations(
    rng: np.random.Generator, length: int, sigma: float
) -> tuple[np.ndarray, dict]:
    """Innovation noise of scale ``sigma``; a fraction of series is heavy-tailed.

    The Student-t draw is variance-normalised so ``sigma`` keeps its meaning; what
    changes is tail mass, i.e. how often the *normal* process produces spike-like
    excursions that a naive detector confuses with anomalies.
    """
    if rng.random() < P_HEAVY_TAIL:
        df = float(rng.uniform(*HEAVY_TAIL_DF))
        scale = sigma / np.sqrt(df / (df - 2.0))
        draw = rng.standard_t(df, length) * scale
        # Truncate the tail by REDRAWING - a hard clip would pile identical values at
        # the bound, an artifact of its own.
        bound = HEAVY_TAIL_CLIP * sigma
        for _ in range(50):
            over = np.abs(draw) > bound
            if not over.any():
                break
            draw[over] = rng.standard_t(df, int(over.sum())) * scale
        np.clip(draw, -bound, bound, out=draw)  # safety net; redraws all but ensure it
        return draw, {"noise": "student_t", "noise_df": round(df, 2)}
    return rng.normal(0.0, sigma, length), {"noise": "gaussian"}


def _colored_noise(
    rng: np.random.Generator, length: int, sigma: float
) -> tuple[np.ndarray, dict]:
    """Additive noise under deterministic bases, coloured with an AR(1) filter.

    Real residuals around a trend or a seasonal pattern are autocorrelated; iid noise
    makes those pools measurably easier than the monitoring data they stand in for.
    """
    phi = float(rng.uniform(*NOISE_PHI_RANGE))
    innov, nparams = _innovations(rng, length + AR_BURN_IN, sigma)
    noise = lfilter([1.0], [1.0, -phi], innov)[AR_BURN_IN:]
    return noise, {"noise_phi": round(phi, 3), **nparams}


def _base_white_noise(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    sigma = float(rng.uniform(0.5, 1.5))
    noise, nparams = _innovations(rng, length, sigma)
    return noise, {"sigma": sigma, **nparams}


def _sample_ar1_phi(rng: np.random.Generator) -> float:
    """Draw a stationary AR(1) coefficient, avoiding the near-unit-root regime."""
    sign = -1.0 if rng.random() < 0.2 else 1.0
    return float(rng.uniform(0.3, 0.9)) * sign


def _sample_ar2_phi(rng: np.random.Generator) -> tuple[float, float]:
    """Rejection-sample AR(2) coefficients from the stationarity triangle.

    The margin keeps roots off the unit circle; without it a pool advertised as stationary
    would contain near-random-walk draws.
    """
    margin = 0.05
    for _ in range(100):
        phi1 = float(rng.uniform(-1.6, 1.6))
        phi2 = float(rng.uniform(-0.9, 0.6))
        if abs(phi2) < 1.0 - margin and phi1 + phi2 < 1.0 - margin and phi2 - phi1 < 1.0 - margin:
            return phi1, phi2
    return 0.6, -0.3  # deterministic stationary fallback


def _ar_filter(
    rng: np.random.Generator, length: int, phis: Sequence[float], sigma: float
) -> tuple[np.ndarray, dict]:
    """Draw an AR(p) realisation with the burn-in transient removed.

    ``lfilter`` starts from zero initial conditions, so the leading samples are
    under-dispersed - which would read as a variance anomaly at the head of every series.
    """
    innovations, nparams = _innovations(rng, length + AR_BURN_IN, sigma)
    denominator = np.concatenate(([1.0], -np.asarray(phis, dtype=np.float64)))
    return lfilter([1.0], denominator, innovations)[AR_BURN_IN:], nparams


def _base_ar1(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    phi = _sample_ar1_phi(rng)
    sigma = float(rng.uniform(0.4, 1.0))
    values, nparams = _ar_filter(rng, length, [phi], sigma)
    return values, {"phi": phi, "sigma": sigma, **nparams}


def _base_ar2(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    phi1, phi2 = _sample_ar2_phi(rng)
    sigma = float(rng.uniform(0.4, 1.0))
    values, nparams = _ar_filter(rng, length, [phi1, phi2], sigma)
    return values, {"phi1": phi1, "phi2": phi2, "sigma": sigma, **nparams}


def _base_linear_trend(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    sigma = float(rng.uniform(0.3, 0.8))
    # Parameterised by total rise, so the slope stays meaningful at any length.
    total_rise = float(rng.uniform(4.0, 25.0)) * (1.0 if rng.random() < 0.5 else -1.0)
    slope = total_rise / length
    t = np.arange(length, dtype=np.float64)
    noise, nparams = _colored_noise(rng, length, sigma)
    return slope * t + noise, {
        "slope": slope,
        "total_rise": total_rise,
        "sigma": sigma,
        **nparams,
    }


def _draw_period(rng: np.random.Generator, length: int) -> int:
    """Draw a seasonal period that fits several times into the series."""
    max_period = max(12, min(200, length // 8))
    return int(rng.integers(12, max_period + 1))


def _base_seasonal_sine(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    period = _draw_period(rng, length)
    amp = float(rng.uniform(1.0, 3.5))
    sigma = float(rng.uniform(0.2, 0.7))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    # A second harmonic keeps the waveform from being a textbook sine.
    harmonic = float(rng.uniform(0.0, 0.35))
    t = np.arange(length, dtype=np.float64)
    seasonal = amp * np.sin(2.0 * np.pi * t / period + phase) + amp * harmonic * np.sin(
        4.0 * np.pi * t / period + phase
    )
    params = {"period": period, "amp": amp, "phase": phase, "harmonic": harmonic, "sigma": sigma}

    if rng.random() < P_SECOND_SEASON:
        # Nested seasonality (a daily cycle inside a weekly one): a longer, weaker
        # oscillation the seasonality-warp anomaly deliberately leaves untouched.
        period2 = period * float(rng.uniform(4.0, 8.0))
        amp2 = amp * float(rng.uniform(0.2, 0.5))
        phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
        seasonal = seasonal + amp2 * np.sin(2.0 * np.pi * t / period2 + phase2)
        params.update(period2=round(period2, 2), amp2=round(amp2, 3))
    if rng.random() < P_AMP_MODULATION:
        # Slow amplitude drift: "normal" is allowed to breathe, so a detector cannot
        # equate any envelope change with an anomaly.
        depth = float(rng.uniform(0.10, 0.30))
        cycles = float(rng.uniform(1.0, 3.0))
        phase_m = float(rng.uniform(0.0, 2.0 * np.pi))
        seasonal = seasonal * (1.0 + depth * np.sin(2.0 * np.pi * cycles * t / length + phase_m))
        params.update(mod_depth=round(depth, 3), mod_cycles=round(cycles, 2))

    noise, nparams = _colored_noise(rng, length, sigma)
    return seasonal + noise, {**params, **nparams}


def _base_trend_seasonal(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    seasonal, params = _base_seasonal_sine(rng, length)
    total_rise = float(rng.uniform(4.0, 20.0)) * (1.0 if rng.random() < 0.5 else -1.0)
    slope = total_rise / length
    t = np.arange(length, dtype=np.float64)
    return seasonal + slope * t, {**params, "slope": slope, "total_rise": total_rise}


BASE_PROCESSES: dict[str, Callable[[np.random.Generator, int], tuple[np.ndarray, dict]]] = {
    "white_noise": _base_white_noise,
    "ar1": _base_ar1,
    "ar2": _base_ar2,
    "linear_trend": _base_linear_trend,
    "seasonal_sine": _base_seasonal_sine,
    "trend_seasonal": _base_trend_seasonal,
}


# --------------------------------------------------------------------------------------
# Scale estimation and windowing
# --------------------------------------------------------------------------------------

def local_scale(values: np.ndarray) -> float:
    """Robust innovation scale: ``MAD(diff(x)) * 1.4826 / sqrt(2)``.

    Differencing removes trend and attenuates seasonality; the MAD resists existing
    structure. This is the unit ``severity`` is measured in.
    """
    if values.size < 2:
        return 1.0
    d = np.diff(values)
    mad = float(np.median(np.abs(d - np.median(d))))
    scale = 1.4826 * mad / np.sqrt(2.0)  # MAD -> sigma for a normal distribution
    if scale > 1e-9:
        return scale
    # Degenerate (near-constant) series: fall back to the marginal spread.
    fallback = float(np.std(values))
    return fallback if fallback > 1e-9 else 1.0


def taper(duration: int, ramp: int) -> np.ndarray:
    """Trapezoidal window, exactly zero at both ends.

    Keeps the series continuous at the segment boundaries, so the labelled span covers
    every modified sample and nothing outside it is disturbed.
    """
    ramp = int(np.clip(ramp, 1, max(1, duration // 2)))
    w = np.ones(duration, dtype=np.float64)
    edge = np.linspace(0.0, 1.0, ramp, endpoint=False) if ramp > 1 else np.zeros(1)
    w[:ramp] = edge
    w[duration - ramp:] = edge[::-1]
    return w


def _draw_segment(
    rng: np.random.Generator, length: int, duration: int,
    forbidden: tuple = (), margin: int = EVENT_MARGIN,
) -> Optional[tuple[int, int]]:
    """Place a segment of ``duration`` samples wholly inside the series.

    ``forbidden`` lists spans already occupied by earlier events; a placement must keep
    ``margin`` clean samples away from each of them so two events never read as one.
    With no forbidden spans the first draw is always accepted. Returns ``None`` when no
    placement fits.
    """
    duration = int(np.clip(duration, 4, length))
    for _ in range(30):
        start = int(rng.integers(0, length - duration + 1))
        end = start + duration
        if all(end + margin <= s or start >= e + margin for s, e in forbidden):
            return start, end
    return None


def _whiten_poly(base_type: str, base_params: dict) -> tuple[float, ...]:
    """AR polynomial that turns the series' stochastic part into innovations.

    An additive perturbation passes through the same whitening filter as the data, so
    its detectability is judged where an optimal detector would judge it - in
    innovation space, not against the marginal spread the autocorrelation inflates.
    """
    if base_type == "ar1":
        return (1.0, -float(base_params["phi"]))
    if base_type == "ar2":
        return (1.0, -float(base_params["phi1"]), -float(base_params["phi2"]))
    phi = float(base_params.get("noise_phi", 0.0))
    return (1.0, -phi) if phi else (1.0,)


def _certified_magnitude(
    shape: np.ndarray, magnitude: float, ctx: "InjectionContext"
) -> tuple[float, float]:
    """Floor an additive magnitude at the certification line.

    ``shape`` is the unit-magnitude profile about to be added. The magnitude comes in
    already sized as severity * sigma_vis - proportional to the series' visible band -
    and is only RAISED here, when the whitened matched-filter z against the series'
    known AR polynomial and innovation scale falls short of SEGMENT_SNR_FLOOR. SNR is
    linear in the magnitude, so the floor is exact. Returns (magnitude, achieved z).
    """
    unit = float(np.linalg.norm(np.convolve(shape, np.asarray(ctx.whiten))))
    unit /= max(ctx.sigma_innov, 1e-12)
    if unit <= 1e-12:
        return magnitude, 0.0
    if CERTIFY_DETECTABILITY:
        magnitude = max(magnitude, SEGMENT_SNR_FLOOR / unit)
    return magnitude, magnitude * unit


def _robust_sd(x: np.ndarray) -> float:
    """Innovation-scale spread of a short span, insensitive to level and trend."""
    if x.size < 3:
        return 0.0
    return float(np.std(np.diff(x))) / math.sqrt(2.0)


def visual_profile(
    values: np.ndarray, period: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Visible geometry of a series: slow baseline, residual band, and its extremes.

    The baseline is a rolling median whose window exceeds the seasonal period, so the
    seasonal swing stays IN the residual while trend and slow wander are taken OUT.
    Returns ``(baseline, residual, sigma_vis, reach, tail_step)``:

    sigma_vis  robust spread (1.4826 * MAD) of the residual - the band a plot shows
               and the unit every severity is measured in.
    reach      q99.5 of |residual| - how far the series ITSELF strays from its
               baseline; a point outlier must land beyond it.
    tail_step  q99.9 - q99.5 of |residual| - how fast the tail keeps going past the
               reach; the outlier margin scales with it.
    """
    n = values.size
    win = max(101, 2 * int(period) + 1) if period else 101
    win = min(win, max(3, n // 3))
    win += 1 - win % 2  # median_filter wants an odd window
    baseline = median_filter(values, size=win, mode="nearest")
    residual = values - baseline
    centered = residual - float(np.median(residual))
    mad = float(np.median(np.abs(centered)))
    sigma_vis = 1.4826 * mad if mad > 1e-9 else max(float(np.std(residual)), 1e-9)
    magnitudes = np.abs(centered)
    reach = float(np.quantile(magnitudes, 0.995))
    tail_step = max(float(np.quantile(magnitudes, 0.999)) - reach, 0.0)
    reach = max(reach, 1.2 * sigma_vis)  # degenerate guard for near-constant residuals
    return baseline, residual, sigma_vis, reach, tail_step


def _baseline_wander(baseline: np.ndarray, horizon: int) -> float:
    """How far the slow baseline naturally moves over ``horizon`` samples, detrended.

    A mean step or drift smaller than the baseline's own excursions at the same time
    scale is camouflage, not an anomaly; this is the masking scale it must beat. The
    deterministic trend is fitted out first - a smooth trend does not mask a sharp
    step, only the stochastic wander around it does.
    """
    n = baseline.size
    if n < 8:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, baseline, 1)
    detrended = baseline - (slope * t + intercept)
    h = int(np.clip(horizon, 1, n - 1))
    moves = np.abs(detrended[h:] - detrended[:-h])
    return float(np.quantile(moves, 0.995)) if moves.size else 0.0


# --------------------------------------------------------------------------------------
# Anomaly injectors
# --------------------------------------------------------------------------------------
# Signature: (rng, values, ctx) -> list[AnomalySegment]; ``values`` is modified in place.
# Labels are derived from the returned segments, never written by the injector.

@dataclass(frozen=True)
class InjectionContext:
    """Everything an injector is allowed to know about the series it perturbs."""

    scale: float
    severity: float
    target_points: int
    base_type: str
    base_params: dict
    # Spans already occupied by earlier events of the same series.
    forbidden: tuple = ()
    # Whitening data for detectability certification (see the CONFIGURATION block).
    whiten: tuple = (1.0,)
    sigma_innov: float = 1.0
    noise: str = "gaussian"
    noise_df: Optional[float] = None
    # Visible geometry of the CLEAN series (see visual_profile); sizes every additive
    # magnitude. baseline/residual are full-length arrays, never mutated by injectors.
    sigma_vis: float = 1.0
    baseline: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None
    reach: float = 1.0
    tail_step: float = 0.0
    # The severity range this family draws from; lets an injector map the drawn
    # severity onto its own natural dial (e.g. the point margin).
    severity_span: tuple[float, float] = (1.0, 1.0)

    @property
    def period(self) -> Optional[int]:
        p = self.base_params.get("period")
        return int(p) if p else None


def _inject_point(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Isolated outliers anchored a fixed distance OUTSIDE the series' own band.

    Every spike lands at ``baseline +- d``, ``d = reach + margin``: past the farthest
    the series itself strays, by a severity-controlled margin that widens with the
    residual's tail. Anchoring to the baseline makes the displacement exact - added on
    top of the current sample, a +3 sigma spike on a -3 sigma sample lands at the mean.
    A sign that would drop the spike where the sample already sits is flipped to the
    far side, so a labelled point is never a relabelling of a natural extreme.
    Positions are drawn in a compressed index space and re-expanded, which enforces the
    minimum spacing exactly; certification treats the whole pattern as one event.
    """
    length = values.size
    gap = POINT_MIN_SPACING
    edge = int(min(POINT_EDGE_MARGIN, max(0, (length - (gap + 1)) // 4)))
    usable = length - 2 * edge
    max_points = max(1, (usable - 1) // (gap + 1))
    n_points = int(min(max(1, ctx.target_points), max_points))

    free = usable - (n_points - 1) * gap
    base = np.sort(rng.choice(free, size=n_points, replace=False))
    positions = edge + base + np.arange(n_points) * gap

    signs = np.where(rng.random(n_points) < 0.5, -1.0, 1.0)
    s_lo, s_hi = ctx.severity_span
    u = (ctx.severity - s_lo) / (s_hi - s_lo) if s_hi > s_lo else 0.5
    u = float(np.clip(u, 0.0, 1.0))
    step = max(ctx.tail_step, 0.35 * ctx.sigma_vis)
    d = ctx.reach + 0.25 * ctx.sigma_vis + (0.25 + 1.0 * u) * step
    # Upward-only jitter (d stays the guaranteed minimum): without it, same-sign spikes
    # on a flat baseline all sit on one exact horizontal - a memorisable artifact.
    displacements = d + rng.uniform(0.0, 0.5, n_points) * ctx.sigma_vis

    r_pos = ctx.residual[positions]
    signs = np.where(np.abs(signs * displacements - r_pos) < ctx.sigma_vis, -signs, signs)

    def aggregate_z(delta: np.ndarray) -> float:
        full = np.zeros(length)
        full[positions] = delta
        z = float(np.linalg.norm(np.convolve(full, np.asarray(ctx.whiten))))
        return z / max(ctx.sigma_innov, 1e-12)

    delta = signs * displacements - r_pos
    snr = aggregate_z(delta)
    if CERTIFY_DETECTABILITY:
        for _ in range(12):
            if snr >= SEGMENT_SNR_FLOOR:
                break
            displacements = displacements * 1.2
            d *= 1.2
            delta = signs * displacements - r_pos
            snr = aggregate_z(delta)

    values[positions] += delta
    shared = {"severity_effective": d / ctx.sigma_vis, "snr": snr}
    return [
        AnomalySegment("point", int(p), int(p) + 1, ctx.severity,
                       {"sign": int(s), "displacement": float(di), **shared})
        for p, s, di in zip(positions, signs, displacements)
    ]


def _alien_waveform(rng: np.random.Generator, duration: int) -> np.ndarray:
    """A unit-amplitude waveform that no base process in this module can produce.

    The period is drawn in SAMPLES (12-40, >= 2.5 cycles): fast enough that no rolling
    baseline absorbs the waveform as level steps, slow enough that its geometry stays
    legible instead of blending into the noise.
    """
    shape = ("square", "triangle", "sawtooth")[int(rng.integers(3))]
    period = float(np.clip(rng.uniform(12.0, 40.0), 4.0, duration / 2.5))
    cycles = duration / period
    frac = np.linspace(0.0, cycles, duration, endpoint=False) % 1.0
    if shape == "square":
        return np.where(frac < 0.5, 1.0, -1.0)
    if shape == "triangle":
        return 4.0 * np.abs(frac - 0.5) - 1.0
    return 2.0 * frac - 1.0  # sawtooth


def _inject_group(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Shapelet anomaly: an alien deterministic waveform is laid over the segment.

    Unlike ``variance`` (resamples noise) and ``level_shift`` (moves the mean), here the
    *shape* of the subsequence stops belonging to the generating process.
    """
    seg = _draw_segment(rng, values.size, ctx.target_points, ctx.forbidden)
    if seg is None:
        return []
    start, end = seg
    duration = end - start
    shape = taper(duration, max(2, duration // 10)) * _alien_waveform(rng, duration)
    # Severity measures the PEAK-TO-PEAK span in sigma_vis units, so the amplitude is
    # half of it. Two-sided adaptation: floored at half the series' reach (a smaller
    # shapelet drowns in a wide band), capped just above it (a caricature otherwise).
    magnitude = ctx.severity * ctx.sigma_vis / 2.0
    if CERTIFY_DETECTABILITY:
        magnitude = min(max(magnitude, 0.5 * ctx.reach), 0.5 * ctx.reach + 0.75 * ctx.sigma_vis)
    magnitude, snr = _certified_magnitude(shape, magnitude, ctx)
    values[start:end] += magnitude * shape
    # The taper is exactly zero at both segment endpoints, so those two samples are
    # unmodified; the label covers the perturbed support only.
    return [AnomalySegment("group", start + 1, end - 1, ctx.severity, {
        "duration": duration - 2, "snr": snr,
        "severity_effective": 2.0 * magnitude / ctx.sigma_vis,
    })]


def _inject_level_shift(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Sustained mean step with deliberately sharp (2-sample) edges."""
    seg = _draw_segment(rng, values.size, ctx.target_points, ctx.forbidden)
    if seg is None:
        return []
    start, end = seg
    duration = end - start
    shape = taper(duration, 2)
    magnitude = ctx.severity * ctx.sigma_vis
    if CERTIFY_DETECTABILITY and ctx.baseline is not None:
        # Two-sided: capped just above the series' reach (clear the band, not dwarf
        # it), floored against the baseline's own wander at this horizon (slow wander
        # is camouflage for a mean step); legibility wins over the cap, 3x drawn max.
        drawn = magnitude
        magnitude = min(magnitude, max(ctx.reach + 0.9 * ctx.sigma_vis, 2.0 * ctx.sigma_vis))
        wander = _baseline_wander(ctx.baseline, duration)
        magnitude = max(magnitude, min(0.75 * wander + 0.5 * ctx.sigma_vis, 3.0 * drawn))
    magnitude, snr = _certified_magnitude(shape, magnitude, ctx)
    shift = magnitude * (1.0 if rng.random() < 0.5 else -1.0)
    values[start:end] += shape * shift
    return [AnomalySegment("level_shift", start + 1, end - 1, ctx.severity, {
        "shift": shift, "duration": duration - 2, "snr": snr,
        "severity_effective": magnitude / ctx.sigma_vis,
    })]


def _inject_variance(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Mean-preserving inflation of local dispersion.

    The local level is estimated with a centred moving average, so the routine is safe on
    trending bases; fluctuations around it are resampled at an inflated scale.
    """
    seg = _draw_segment(rng, values.size, ctx.target_points, ctx.forbidden)
    if seg is None:
        return []
    start, end = seg
    duration = end - start
    w = taper(duration, max(2, duration // 8))

    segment = values[start:end]
    window = max(3, min(duration // 4, 51) | 1)  # odd length keeps the average centred
    kernel = np.ones(window) / window
    level = np.convolve(np.pad(segment, window // 2, mode="edge"), kernel, mode="valid")[:duration]

    # Dispersion-only change: the existing fluctuation is SCALED, not resampled.
    # Resampling with white noise silently added a second mechanism (decorrelation) on
    # AR bases and swapped the tail law; scaling preserves waveform, autocorrelation
    # and tails, and touches exactly one property - the local dispersion.
    fluctuation = segment - level
    if _robust_sd(segment) < 0.05 * ctx.scale:
        return []  # degenerate placement: nothing to inflate at this spot

    inflation = 1.0 + ctx.severity * 0.5
    if CERTIFY_DETECTABILITY and ctx.residual is not None and ctx.residual.size > duration:
        # An inflation inside the series' own rolling-scale swing is camouflage: floor
        # the ratio just above its q99.5 at the same window, capped at 2.5x the drawn.
        r = ctx.residual
        window_mean = np.convolve(r * r, np.ones(duration) / duration, mode="valid")
        rolling = np.sqrt(np.maximum(window_mean, 0.0))
        med = float(np.median(rolling))
        if med > 1e-9:
            natural = float(np.quantile(rolling, 0.995)) / med
            inflation = max(inflation, min(1.2 * natural, 2.5 * (1.0 + ctx.severity * 0.5)))
    n_eff = float(np.sum(w ** 2))
    if CERTIFY_DETECTABILITY:
        # z ~ sqrt(n_eff/2) * ln(r) for a dispersion ratio r observed on n_eff points;
        # the safety cap keeps a degenerate short placement from demanding a blow-up.
        r_min = math.exp(SEGMENT_SNR_FLOOR / math.sqrt(max(n_eff, 2.0) / 2.0))
        inflation = min(max(inflation, r_min), max(6.0, r_min))
    snr = math.sqrt(n_eff / 2.0) * math.log(inflation)

    values[start:end] = level + fluctuation * (1.0 + w * (inflation - 1.0))
    # severity_effective maps the applied ratio back onto the severity dial
    # (inflation = 1 + severity/2), so it is comparable with the drawn severity.
    return [AnomalySegment("variance", start + 1, end - 1, ctx.severity, {
        "inflation": inflation, "duration": duration - 2, "snr": snr,
        "severity_effective": 2.0 * (inflation - 1.0),
    })]


def _inject_trend(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Transient linear drift that peaks inside the segment and returns to zero by its end.

    Returning to zero keeps the label honest: a drift left hanging at the edge would put an
    unlabelled discontinuity right after the labelled span.
    """
    seg = _draw_segment(rng, values.size, ctx.target_points, ctx.forbidden)
    if seg is None:
        return []
    start, end = seg
    duration = end - start

    knee = int(np.clip(int(duration * float(rng.uniform(0.6, 0.85))), 1, duration - 2))
    profile = np.empty(duration, dtype=np.float64)
    profile[:knee] = np.linspace(0.0, 1.0, knee, endpoint=False)
    profile[knee:] = np.linspace(1.0, 0.0, duration - knee)  # endpoint included -> ends at 0

    magnitude = ctx.severity * ctx.sigma_vis
    if CERTIFY_DETECTABILITY and ctx.baseline is not None:
        # Same two-sided adaptation as level_shift: reach cap above, wander floor
        # below, legibility winning over the cap.
        drawn = magnitude
        magnitude = min(magnitude, max(ctx.reach + 0.9 * ctx.sigma_vis, 2.0 * ctx.sigma_vis))
        wander = _baseline_wander(ctx.baseline, duration)
        magnitude = max(magnitude, min(0.75 * wander + 0.5 * ctx.sigma_vis, 3.0 * drawn))
    magnitude, snr = _certified_magnitude(profile, magnitude, ctx)
    drift = magnitude * (1.0 if rng.random() < 0.5 else -1.0)
    values[start:end] += profile * drift
    return [AnomalySegment("trend", start + 1, end - 1, ctx.severity, {
        "peak_drift": drift, "duration": duration - 2, "snr": snr,
        "severity_effective": magnitude / ctx.sigma_vis,
    })]


def _inject_seasonality(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Distort the seasonal component by warping its period in place.

    The original oscillation is subtracted and replaced at a perturbed period - a genuine
    frequency change, not an extra sine on top. Non-seasonal bases get an alien
    oscillation instead, since there is nothing to warp.
    """
    period = ctx.period
    direction_first = 1.0 if rng.random() < 0.5 else -1.0

    target_points = ctx.target_points
    warp = None
    if period is not None:
        # The warp must accumulate at least one FULL cycle of discrepancy over the
        # span, or the eye (and a spectral detector) sees a phase wobble, not a
        # frequency change. Given the drawn warp, that fixes a minimum duration; when
        # the series is too short for it, the warp is strengthened instead.
        warp = float(np.clip(1.0 + ctx.severity * 0.12 * direction_first, 0.4, 2.2))
        max_dur = max(2 * period + 2, int(0.3 * values.size))
        if CERTIFY_DETECTABILITY:
            need = int(math.ceil(period / abs(1.0 - 1.0 / warp))) + 2
            if need > max_dur:
                # strengthen the warp until one full cycle of discrepancy fits
                ratio = period / max(max_dur - 2, 1)
                warp = 1.0 / (1.0 - ratio) if direction_first > 0 else 1.0 / (1.0 + ratio)
                warp = float(np.clip(warp, 0.4, 2.2))
                need = int(math.ceil(period / max(abs(1.0 - 1.0 / warp), 1e-9))) + 2
            target_points = max(target_points, min(need, max_dur))

    seg = _draw_segment(rng, values.size, target_points, ctx.forbidden)
    if seg is None:
        return []
    start, end = seg
    duration = end - start
    w = taper(duration, max(2, duration // 8))
    t = np.arange(start, end, dtype=np.float64)
    direction = direction_first

    if period is None:
        # At least two alien cycles must fit, or the "oscillation" reads as a bump.
        alien_period = min(float(rng.uniform(8.0, 40.0)), duration / 2.0)
        shape = w * np.sin(2.0 * np.pi * t / alien_period)
        # Symmetric oscillation: severity is the peak-to-peak span in sigma_vis units,
        # with the same two-sided reach adaptation as the group waveform.
        magnitude = ctx.severity * ctx.sigma_vis / 2.0
        if CERTIFY_DETECTABILITY:
            magnitude = min(max(magnitude, 0.5 * ctx.reach), 0.5 * ctx.reach + 0.75 * ctx.sigma_vis)
        magnitude, snr = _certified_magnitude(shape, magnitude, ctx)
        values[start:end] += magnitude * shape
        return [
            AnomalySegment(
                "seasonality", start + 1, end - 1, ctx.severity,
                {"mode": "injected", "alien_period": alien_period, "duration": duration - 2,
                 "snr": snr, "severity_effective": 2.0 * magnitude / ctx.sigma_vis},
            )
        ]

    amp = float(ctx.base_params["amp"])
    phase = float(ctx.base_params["phase"])
    harmonic = float(ctx.base_params.get("harmonic", 0.0))

    def wave(p: float) -> np.ndarray:
        return amp * np.sin(2.0 * np.pi * t / p + phase) + amp * harmonic * np.sin(
            4.0 * np.pi * t / p + phase
        )

    def delta_for(wr: float) -> tuple[np.ndarray, float]:
        d = w * (wave(period * wr) - wave(float(period)))
        z = float(np.linalg.norm(np.convolve(d, np.asarray(ctx.whiten)))) / max(ctx.sigma_innov, 1e-12)
        return d, z

    # The warp is not a free amplitude, so neither floor can be solved in closed form.
    # Escalate the warp deterministically while the discrepancy is short of the
    # certification floor OR of its target visible size; once the warp saturates its
    # clip bounds, scale the discrepancy itself. Certification stays rng-free.
    def visible_span(d: np.ndarray) -> float:
        return float(np.quantile(d, 0.999) - np.quantile(d, 0.001))

    target_span = ctx.severity * ctx.sigma_vis
    delta, snr = delta_for(warp)
    if CERTIFY_DETECTABILITY:
        for _ in range(8):
            if snr >= SEGMENT_SNR_FLOOR and visible_span(delta) >= 0.9 * target_span:
                break
            stronger = float(np.clip(1.0 + (warp - 1.0) * 1.3, 0.4, 2.2))
            if stronger == warp:
                break  # warp saturated; the blend below takes over
            warp = stronger
            delta, snr = delta_for(warp)
    # Visible calibration: the discrepancy spans severity * sigma_vis peak-to-peak like
    # every other symmetric family - a warp the noise buries is scaled up, an over-loud
    # one scaled back down. The blend keeps the frequency-change character: the segment
    # becomes original + blend * (warped - original).
    blend = 1.0
    span = visible_span(delta)
    if span > 1e-9:
        blend = float(np.clip(target_span / span, 0.4, 3.0))
        delta = delta * blend
        snr = snr * blend
    if CERTIFY_DETECTABILITY and 1e-9 < snr < SEGMENT_SNR_FLOOR:
        boost = SEGMENT_SNR_FLOOR / snr
        delta = delta * boost
        blend *= boost
        snr = SEGMENT_SNR_FLOOR

    values[start:end] += delta
    new_period = period * warp
    return [
        AnomalySegment(
            "seasonality", start + 1, end - 1, ctx.severity,
            {"mode": "warped", "period": period, "new_period": new_period,
             "warp": warp, "blend": blend, "duration": duration - 2, "snr": snr,
             "severity_effective": visible_span(delta) / ctx.sigma_vis},
        )
    ]


def _inject_flatline(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Stuck-sensor anomaly: the segment collapses onto a constant held value.

    One of the most common industrial failure modes and mechanically unlike every other
    family here - it *removes* variation instead of adding any. The flat level is the
    segment median, and short edge blends keep the series continuous at the labelled
    boundaries (a dead sensor's edges are sharp, but both cliffs stay inside the label).
    """
    seg = _draw_segment(rng, values.size, ctx.target_points, ctx.forbidden)
    if seg is None:
        return []
    start, end = seg
    duration = end - start
    ramp = max(3, duration // 10)
    w = taper(duration, ramp)

    segment = values[start:end]
    sd_before = _robust_sd(segment)
    if CERTIFY_DETECTABILITY and sd_before < 0.05 * ctx.scale:
        return []  # the base is already flat here - a stuck sensor would be invisible

    level = float(np.median(segment))
    perturbation = w * (level - segment)
    values[start:end] = segment + perturbation

    # Certification is structural: n_core exactly-constant samples on a series whose
    # local noise is alive (the sd guard above) cannot occur naturally - a zero-diff
    # run is the one signature no stochastic base here produces. The recorded z is the
    # whitened matched-filter statistic of the applied perturbation, a conservative
    # lower bound (it does not credit the run-length signature).
    n_core = int(np.sum(w >= 0.999))
    snr = float(np.linalg.norm(np.convolve(perturbation, np.asarray(ctx.whiten))))
    snr /= max(ctx.sigma_innov, 1e-12)
    return [AnomalySegment("flatline", start + 1, end - 1, ctx.severity, {
        "level": level, "duration": duration - 2, "n_core": n_core, "snr": snr,
    })]


ANOMALY_INJECTORS: dict[
    str, Callable[[np.random.Generator, np.ndarray, InjectionContext], list[AnomalySegment]]
] = {
    "point": _inject_point,
    "group": _inject_group,
    "level_shift": _inject_level_shift,
    "variance": _inject_variance,
    "trend": _inject_trend,
    "seasonality": _inject_seasonality,
    "flatline": _inject_flatline,
}


# --------------------------------------------------------------------------------------
# Series assembly
# --------------------------------------------------------------------------------------

def generate_series(index: int, config: PoolConfig, rng: np.random.Generator) -> SeriesResult:
    """Generate one labelled series from its own random stream."""
    lo, hi = config.length_range
    length = int(rng.integers(lo, hi + 1)) if hi > lo else lo

    base_type = config.base_types[int(rng.integers(len(config.base_types)))]
    values, base_params = BASE_PROCESSES[base_type](rng, length)
    values = np.asarray(values, dtype=np.float64)
    # Real metrics live at arbitrary levels (a CPU near 80, a latency near 200);
    # a constant offset costs nothing and catches zero-mean assumptions downstream.
    baseline = float(rng.uniform(*BASELINE_OFFSET))
    values = values + baseline
    base_params = {**base_params, "baseline": round(baseline, 3)}

    labels = np.zeros(length, dtype=np.int8)
    segments: list[AnomalySegment] = []
    anomaly_type = "none"
    severity = 0.0
    target_fraction = 0.0

    if rng.random() < config.anomaly_rate:
        anomaly_type = config.allowed_anomaly_types[int(rng.integers(len(config.allowed_anomaly_types)))]
        f_lo, f_hi = config.anomaly_fraction_range
        if anomaly_type == "point" and CERTIFY_DETECTABILITY:
            f_lo, f_hi = POINT_FRACTION_RANGE
        f_lo, f_hi = ANOMALY_FRACTION_OVERRIDES.get(anomaly_type, (f_lo, f_hi))
        s_lo, s_hi = SEVERITY_RANGE_OVERRIDES.get(anomaly_type, config.severity_range)
        # Log-uniform: real pools span two orders of magnitude of anomaly density, so
        # every order of magnitude deserves equal coverage.
        target_fraction = float(np.exp(rng.uniform(np.log(f_lo), np.log(f_hi))))
        severity = float(rng.uniform(s_lo, s_hi))

        target_points = max(4, int(round(length * target_fraction)))
        # Measured once on the clean series, so magnitudes stay comparable across pools
        # and later events are not sized against an already-perturbed signal.
        scale = local_scale(values)
        baseline_path, residual, vis, reach, tail_step = visual_profile(
            values, base_params.get("period")
        )

        # Per-family duration floor: below it the family's own test statistic has no
        # power and the label would be noise, whatever the magnitude. Seasonality is
        # additionally floored at two full periods - warping half a cycle is a phase
        # wobble, not a frequency change.
        floor_points = 0
        if CERTIFY_DETECTABILITY and anomaly_type != "point":
            floor_points = MIN_EVENT_POINTS.get(anomaly_type, 4)
            period_param = base_params.get("period")
            if anomaly_type == "seasonality" and period_param:
                floor_points = max(floor_points, 2 * int(period_param) + 2)
            if anomaly_type in ("level_shift", "trend") and period_param:
                # Shorter than a couple of cycles, a step or drift rides one wave
                # flank and reads as the wave's own motion or as amplitude
                # modulation, which the clean bases are allowed to have.
                floor_points = max(floor_points, PERIODIC_MIN_PERIODS * int(period_param) + 2)
            floor_points = min(floor_points, length)

        n_events = 1
        if anomaly_type != "point" and rng.random() < P_TWO_EVENTS:
            # Two disjoint events of one family; "point" already scatters its own
            # spikes. Both events must clear the duration floor, or there is one.
            if target_points >= 2 * max(floor_points, 4):
                n_events = 2
        budgets = (
            [target_points]
            if n_events == 1
            else [max(4, int(round(target_points * 0.6))), max(4, int(round(target_points * 0.4)))]
        )

        for event_points in budgets:
            ctx = InjectionContext(
                scale=scale,
                severity=severity,
                target_points=max(event_points, floor_points),
                base_type=base_type,
                base_params=base_params,
                forbidden=tuple((seg.start, seg.end) for seg in segments),
                whiten=_whiten_poly(base_type, base_params),
                sigma_innov=float(base_params.get("sigma", 1.0)),
                noise=str(base_params.get("noise", "gaussian")),
                noise_df=base_params.get("noise_df"),
                sigma_vis=vis,
                baseline=baseline_path,
                residual=residual,
                reach=reach,
                tail_step=tail_step,
                severity_span=(s_lo, s_hi),
            )
            segments.extend(ANOMALY_INJECTORS[anomaly_type](rng, values, ctx))
        if not segments:
            # Every placement failed (or was degenerate): the series IS clean, and its
            # metadata must say so rather than advertise an anomaly that is not there.
            anomaly_type = "none"
            severity = 0.0
            target_fraction = 0.0
        for seg in segments:
            labels[seg.start:seg.end] = 1

    num_anomalies = int(labels.sum())
    return SeriesResult(
        series_id=f"{config.group}__SYNTH__{base_type}-{index:05d}_full",
        values=values,
        labels=labels,
        metadata={
            "series_id": f"{config.group}__SYNTH__{base_type}-{index:05d}_full",
            "length": length,
            "num_point_anomalies": num_anomalies,
            "y_i": int(num_anomalies > 0),
            "is_split": False,
            "original_length": length,
            "source_notes": f"synthetic;base={base_type};anomaly={anomaly_type}",
            "base_type": base_type,
            "anomaly_type": anomaly_type,
            "anomaly_fraction": round(num_anomalies / length, 6),
            "target_fraction": round(target_fraction, 6),
            "severity": round(severity, 4),
            "severity_effective": round(
                min((s.detail.get("severity_effective", severity) for s in segments), default=0.0), 4
            ),
            "detect_z": round(min((s.detail.get("snr", 0.0) for s in segments), default=0.0), 2),
            "num_segments": len(segments),
            "base_params": json.dumps({k: _jsonable(v) for k, v in base_params.items()}),
            "segments": json.dumps([s.as_dict() for s in segments]),
        },
    )


# --------------------------------------------------------------------------------------
# Pool assembly and validation
# --------------------------------------------------------------------------------------

def validate_pool(data: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Assert every invariant the downstream pipeline relies on, before anything is written."""
    missing = {"series_id", "time_index", "value", "label"} - set(data.columns)
    if missing:
        raise AssertionError(f"data is missing columns: {sorted(missing)}")
    missing = {
        "series_id", "length", "num_point_anomalies", "y_i", "is_split", "original_length",
    } - set(meta.columns)
    if missing:
        raise AssertionError(f"metadata is missing columns: {sorted(missing)}")

    for column, expected in (("value", np.float64), ("time_index", np.int64), ("label", np.int8)):
        if data[column].dtype != expected:
            raise AssertionError(f"{column} must be {expected.__name__}, got {data[column].dtype}")

    if meta["series_id"].duplicated().any():
        dupes = meta.loc[meta["series_id"].duplicated(), "series_id"].tolist()[:5]
        raise AssertionError(f"duplicate series_id in metadata, e.g. {dupes}")
    if not np.isfinite(data["value"].to_numpy()).all():
        raise AssertionError("value contains NaN or inf")
    if not data["label"].isin((0, 1)).all():
        raise AssertionError("label must be binary")

    observed = data.groupby("series_id", sort=False).agg(
        obs_length=("time_index", "size"),
        obs_anomalies=("label", "sum"),
        first_index=("time_index", "min"),
        last_index=("time_index", "max"),
    )
    joined = meta.set_index("series_id").join(observed, how="left")

    if joined["obs_length"].isna().any():
        raise AssertionError("metadata references a series_id absent from the data frame")
    if not (joined["obs_length"] == joined["length"]).all():
        raise AssertionError("metadata length disagrees with the number of rows")
    if not (joined["obs_anomalies"] == joined["num_point_anomalies"]).all():
        raise AssertionError("metadata num_point_anomalies disagrees with the labels")
    if not (joined["y_i"] == (joined["obs_anomalies"] > 0).astype(int)).all():
        raise AssertionError("y_i disagrees with the point-wise labels")
    if not (joined["first_index"] == 0).all():
        raise AssertionError("time_index must start at 0 for every series")
    if not (joined["last_index"] == joined["length"] - 1).all():
        raise AssertionError("time_index must be contiguous within every series")


def generate_pool(config: PoolConfig, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate, validate and persist one pool."""
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== %s: generating %d series ===", config.group, config.num_series)
    # crc32, not hash(): the builtin hash of a str is randomised per interpreter run and
    # would silently break reproducibility across processes.
    root = np.random.SeedSequence(config.seed, spawn_key=(zlib.crc32(config.group.encode()),))
    streams = root.spawn(config.num_series)

    results: list[SeriesResult] = []
    for i, seed_seq in enumerate(streams):
        results.append(generate_series(i, config, np.random.default_rng(seed_seq)))
        if (i + 1) % 250 == 0:
            logger.info("  ... %d/%d", i + 1, config.num_series)

    lengths = np.fromiter((r.values.size for r in results), dtype=np.int64, count=len(results))
    data = pd.DataFrame(
        {
            "series_id": np.repeat(np.array([r.series_id for r in results], dtype=object), lengths),
            "time_index": np.concatenate([np.arange(n, dtype=np.int64) for n in lengths]),
            "value": np.concatenate([r.values for r in results]).astype(np.float64),
            "label": np.concatenate([r.labels for r in results]).astype(np.int8),
        }
    )
    meta = pd.DataFrame([r.metadata for r in results])

    validate_pool(data, meta)

    # Grouped evaluation split, mirroring the real pools. Synthetic series are mutually
    # independent, so the "group" is the series itself; the keyed hash keeps the
    # assignment stable under regeneration.
    meta["split"] = [
        "dev" if (zlib.crc32(f"{config.seed}|split|{sid}".encode()) & 0xFFFFFFFF) / 2**32 < DEV_FRACTION
        else "test"
        for sid in meta["series_id"]
    ]

    data.to_parquet(output_dir / f"{config.group}.parquet", index=False)
    meta.to_parquet(output_dir / f"{config.group}_metadata.parquet", index=False)
    manifest = {
        **config.to_manifest(),
        "num_rows": int(len(data)),
        "anomalous_series": int(meta["y_i"].sum()),
        "mean_anomaly_fraction": round(float(meta["anomaly_fraction"].mean()), 6),
    }
    (output_dir / f"{config.group}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _log_summary(config.group, meta)
    logger.info("Saved to %s/%s.parquet", output_dir, config.group)
    return data, meta


def _log_summary(group: str, meta: pd.DataFrame) -> None:
    anomalous = meta[meta["y_i"] == 1]
    logger.info("=== %s COMPLETE ===", group)
    logger.info("Series: %d | anomalous: %d (%.1f%%)", len(meta), len(anomalous), 100 * meta["y_i"].mean())
    logger.info(
        "Length min/mean/max: %d / %.1f / %d",
        meta["length"].min(), meta["length"].mean(), meta["length"].max(),
    )
    if not anomalous.empty:
        logger.info(
            "Anomalous-point fraction (dirty series) min/mean/max: %.4f / %.4f / %.4f",
            anomalous["anomaly_fraction"].min(),
            anomalous["anomaly_fraction"].mean(),
            anomalous["anomaly_fraction"].max(),
        )
        logger.info("Anomaly types: %s", anomalous["anomaly_type"].value_counts().to_dict())
        if "detect_z" in anomalous.columns:
            logger.info(
                "Certified detectability z (weakest event) min/median: %.2f / %.2f",
                anomalous["detect_z"].min(), anomalous["detect_z"].median(),
            )
    logger.info("Base processes: %s", meta["base_type"].value_counts().to_dict())


def main() -> None:
    """Build every pool listed in GROUPS_TO_BUILD."""
    for group in GROUPS_TO_BUILD:
        generate_pool(pool_config(group), OUTPUT_DIR)


if __name__ == "__main__":
    main()
