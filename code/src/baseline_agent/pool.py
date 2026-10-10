"""Pool storage for the raw-retrieval baseline agent.

A pool is a spec-01 parquet file with the flattened columns ``series_id``, ``time_index``,
``value`` and ``label``. This module loads the first three and exposes them read-only.
The ``label`` column is never read - not dropped after loading, never loaded - so nothing
in this package can hand ground truth to the agent by accident.

Series keep the order in which they first appear in the file; :meth:`Pool.reordered`
gives a seeded permutation of that order for runs that should not walk the file from the
top (spec-01 files are grouped by source, see the task README).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REQUIRED_COLUMNS: tuple[str, ...] = ("series_id", "time_index", "value")


@dataclass(frozen=True)
class SeriesData:
    """One univariate series: two parallel arrays, nothing derived."""

    series_id: str
    time_index: np.ndarray
    value: np.ndarray

    def __len__(self) -> int:
        return int(self.value.shape[0])

    def problems(self) -> list[str]:
        """Integrity violations that make the series unusable; empty when it is sound.

        Checked at retrieval time rather than at load time, so a pool with one broken
        series still loads and the agent gets a readable error for that one series
        (spec 02, FR1).
        """
        out: list[str] = []
        n_values = int(self.value.shape[0])
        n_index = int(self.time_index.shape[0])
        if n_values == 0:
            out.append("series has no points")
        if n_index != n_values:
            out.append(f"time_index has {n_index} entries but value has {n_values}")
        elif n_values:
            if int(self.time_index[0]) != 0:
                out.append(f"time_index starts at {int(self.time_index[0])}, expected 0")
            if n_values > 1 and not bool(np.all(np.diff(self.time_index) == 1)):
                out.append("time_index is not contiguous")
        bad = ~np.isfinite(self.value)
        if n_values and bad.any():
            first = int(np.flatnonzero(bad)[0])
            out.append(f"{int(bad.sum())} non-finite value(s) (NaN or inf), first at position {first}")
        return out


class Pool:
    """Read-only, label-free view of a series pool.

    Build one with :meth:`from_parquet` (the normal path) or :meth:`from_frame` (tests).
    """

    def __init__(self, name: str, series: Sequence[SeriesData], *, source: str | None = None):
        ids = [s.series_id for s in series]
        for sid in ids:
            if not sid or sid != sid.strip():
                raise ValueError(f"pool {name!r}: empty or whitespace-padded series_id {sid!r}")
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate series_id in pool {name!r}: {dupes[:5]}")
        self._name = name
        self._source = source
        self._order: tuple[str, ...] = tuple(ids)
        self._series: dict[str, SeriesData] = {s.series_id: s for s in series}

    # --- construction -----------------------------------------------------------------

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, name: str, *, source: str | None = None) -> "Pool":
        """Build a pool from a flattened frame with the spec-01 columns.

        Any column beyond ``series_id`` / ``time_index`` / ``value`` (``label`` included)
        is ignored. Values must be ``float64`` (the spec-01 dtype) or an integer type,
        which is kept: a narrower float would be widened into digits the file never held.
        Rows of one series keep their file order; a series whose rows are not in index
        order is reported as malformed by :meth:`SeriesData.problems`.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"pool {name!r} lacks required column(s) {missing}; has {list(frame.columns)}")
        if len(frame) == 0:
            raise ValueError(f"pool {name!r} is empty")
        if not pd.api.types.is_integer_dtype(frame["time_index"]):
            raise ValueError(f"pool {name!r}: time_index must be an integer column, got {frame['time_index'].dtype}")
        values_col = frame["value"]
        if pd.api.types.is_integer_dtype(values_col):
            value = values_col.to_numpy(dtype=np.int64)
        elif values_col.dtype == np.float64:
            value = values_col.to_numpy(dtype=np.float64)
        else:
            raise ValueError(
                f"pool {name!r}: value must be a float64 or integer column, got {values_col.dtype}; "
                "cast it explicitly rather than let the loader widen it"
            )
        ids = frame["series_id"].to_numpy()
        if pd.isna(ids).any():
            raise ValueError(f"pool {name!r}: series_id contains missing values")

        codes, uniques = pd.factorize(ids, sort=False)
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        bounds = np.flatnonzero(np.diff(sorted_codes)) + 1
        starts = np.concatenate(([0], bounds))
        stops = np.concatenate((bounds, [len(codes)]))
        time_index = frame["time_index"].to_numpy(dtype=np.int64)[order]
        value = value[order]
        series = [
            SeriesData(str(uniques[k]), time_index[a:b].copy(), value[a:b].copy())
            for k, (a, b) in enumerate(zip(starts, stops))
        ]
        return cls(name, series, source=source)

    @classmethod
    def from_parquet(cls, path: str | Path, *, name: str | None = None, source: str | None = None) -> "Pool":
        """Load a spec-01 pool, reading only the three columns the agent may see.

        ``source`` is what the trace records as the pool's path; it defaults to ``path``
        as given.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"pool file not found: {path}")
        present = pq.read_schema(path).names
        missing = [c for c in REQUIRED_COLUMNS if c not in present]
        if missing:
            raise ValueError(f"{path}: lacks required column(s) {missing}; has {present}")
        table = pq.read_table(path, columns=list(REQUIRED_COLUMNS))
        return cls.from_frame(table.to_pandas(), name or path.stem, source=str(path) if source is None else source)

    # --- access -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> str | None:
        return self._source

    @property
    def series_ids(self) -> list[str]:
        """All series ids in pool order."""
        return list(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, series_id: object) -> bool:
        return series_id in self._series

    def __iter__(self) -> Iterator[str]:
        return iter(self._order)

    def get(self, series_id: str) -> SeriesData:
        """Return a series; raises ``KeyError`` for an unknown id."""
        return self._series[series_id]

    def length_of(self, series_id: str) -> int:
        return len(self._series[series_id])

    def subset(self, series_ids: Iterable[str]) -> "Pool":
        """A pool restricted to the given ids, in this pool's order."""
        keep = set(series_ids)
        unknown = keep - set(self._order)
        if unknown:
            raise KeyError(f"unknown series_id(s): {sorted(unknown)[:5]}")
        series = [self._series[i] for i in self._order if i in keep]
        return Pool(self._name, series, source=self._source)

    def reordered(self, seed: int | None) -> "Pool":
        """The same series in a seeded random order; ``None`` keeps the file order."""
        if seed is None:
            return self
        perm = np.random.default_rng(seed).permutation(len(self._order))
        return Pool(self._name, [self._series[self._order[i]] for i in perm], source=self._source)

    @cached_property
    def fingerprint(self) -> str:
        """SHA-256 over ids, indices, values and their dtype, in pool order - identifies the exact input."""
        h = hashlib.sha256()
        for sid in self._order:
            s = self._series[sid]
            h.update(sid.encode("utf-8"))
            h.update(b"\0")
            h.update(np.ascontiguousarray(s.time_index, dtype="<i8").tobytes())
            h.update(str(s.value.dtype).encode("ascii"))
            h.update(np.ascontiguousarray(s.value, dtype=s.value.dtype.newbyteorder("<")).tobytes())
            h.update(b"\n")
        return h.hexdigest()


def read_split_ids(metadata_path: str | Path, split: str) -> list[str]:
    """Series ids of one evaluation split (``dev`` / ``test``) from a spec-01 metadata file.

    Only ``series_id`` and ``split`` are read; the label column ``y_i`` stays on disk.
    """
    table = pq.read_table(Path(metadata_path), columns=["series_id", "split"])
    frame = table.to_pandas()
    known = sorted(frame["split"].unique().tolist())
    if split not in known:
        raise ValueError(f"unknown split {split!r}; metadata has {known}")
    return frame.loc[frame["split"] == split, "series_id"].astype(str).tolist()
