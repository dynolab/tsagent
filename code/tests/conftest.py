"""Shared fixtures: a tiny in-memory pool and its parquet twin. No LLM is ever contacted."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic_ai import models

from src.baseline_agent.pool import Pool

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
models.ALLOW_MODEL_REQUESTS = False  # any attempt to reach a real provider fails loudly


def _series(sid: str, values: list[float], *, start: int = 0) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "series_id": [sid] * n,
            "time_index": np.arange(start, start + n, dtype=np.int64),
            "value": np.asarray(values, dtype=np.float64),
            "label": np.zeros(n, dtype=np.int8),
        }
    )


def tiny_frame() -> pd.DataFrame:
    """Five series: three normal, one with a spike, one with a frozen stretch, plus a
    label column that the pool must ignore. Values are deterministic."""
    rng = np.random.default_rng(0)
    normal_a = (rng.standard_normal(200) * 0.5 + 10).tolist()
    normal_b = (np.sin(np.arange(300) / 7.0) * 3 + rng.standard_normal(300) * 0.2).tolist()
    normal_c = (np.arange(150) * 0.01 + rng.standard_normal(150) * 0.3).tolist()
    spike = (rng.standard_normal(250) * 0.5).tolist()
    spike[120] = 25.0
    frozen = (rng.standard_normal(220) * 0.7 + 3).tolist()
    frozen[60:110] = [3.25] * 50
    frames = [
        _series("T1__SYNTH__normal-00000_full", normal_a),
        _series("T1__SYNTH__normal-00001_full", normal_b),
        _series("T1__SYNTH__spike-00002_full", spike),
        _series("T1__SYNTH__normal-00003_full", normal_c),
        _series("T1__SYNTH__frozen-00004_full", frozen),
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame.loc[frame["series_id"].str.contains("spike") & (frame["time_index"] == 120), "label"] = 1
    return frame


@pytest.fixture
def tiny_pool() -> Pool:
    return Pool.from_frame(tiny_frame(), "T1")


@pytest.fixture
def tiny_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "T1.parquet"
    tiny_frame().to_parquet(path, index=False)
    return path


@pytest.fixture
def malformed_pool() -> Pool:
    """A pool whose second series has a NaN and whose third has a gap in time_index."""
    good = _series("M__a_full", [1.0, 2.0, 3.0, 2.0, 1.0])
    nan = _series("M__nan_full", [1.0, float("nan"), 3.0])
    gap = _series("M__gap_full", [1.0, 2.0, 3.0])
    gap.loc[gap.index[-1], "time_index"] = 5
    return Pool.from_frame(pd.concat([good, nan, gap], ignore_index=True), "M")
