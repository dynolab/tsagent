"""The two tools of the baseline agent, as pure functions over a :class:`Pool`.

Both return text: a JSON document for a successful call, a plain error sentence otherwise.
Returning text instead of raising is what spec 02 asks for - the LLM must be able to read
and reason about a failure - and it also pins the exact wire format, so the size of what
the model sees can be measured to the character and replayed byte for byte.

The agent-facing wrappers that add timing, trace recording and id aliasing live in
``agent.py``; the functions here are what the tests exercise directly.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .pool import Pool, SeriesData

ERROR_PREFIX = "Error:"


def is_error(text: str) -> bool:
    return text.startswith(ERROR_PREFIX)


def dumps_payload(payload: dict[str, Any]) -> str:
    """Compact, deterministic JSON: no whitespace, keys in insertion order, NaN rejected."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def series_payload(series: SeriesData, *, value_precision: int | None = None, label: str | None = None) -> dict[str, Any]:
    """The ``get_series`` document: ``series_id``, ``length`` and the two raw arrays.

    Integer-valued pools stay integers; ``value_precision`` rounds float values to that many
    decimals - a lossy option that is off by default, since FR1 asks for the raw series.
    ``label`` is the id the model knows the series by (an alias, when aliasing is on).
    """
    values = series.value.tolist()
    if value_precision is not None and series.value.dtype.kind == "f":
        values = [round(v, value_precision) for v in values]
    return {
        "series_id": label if label is not None else series.series_id,
        "length": len(series),
        "time_index": series.time_index.tolist(),
        "value": values,
    }


def pool_meta_data(pool: Pool, *, series_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """The ``get_pool_meta_data`` document. Extensible with more pool-level fields (FR2)."""
    return {
        "pool": pool.name,
        "n_series": len(pool),
        "series_ids": list(series_ids) if series_ids is not None else pool.series_ids,
    }


def unknown_id_error(series_id: str) -> str:
    return f"{ERROR_PREFIX} unknown series_id {series_id!r}. Call get_pool_meta_data() for the list of valid ids."


def get_series(pool: Pool, series_id: str, *, value_precision: int | None = None, label: str | None = None) -> str:
    """Full raw series as JSON text, or an ``Error: ...`` sentence (FR1)."""
    if not isinstance(series_id, str) or not series_id.strip():
        return f"{ERROR_PREFIX} series_id must be a non-empty string."
    try:
        series = pool.get(series_id)
    except KeyError:
        return unknown_id_error(label if label is not None else series_id)
    problems = series.problems()
    if problems:
        shown = label if label is not None else series_id
        return f"{ERROR_PREFIX} series {shown!r} is malformed and cannot be returned: " + "; ".join(problems) + "."
    return dumps_payload(series_payload(series, value_precision=value_precision, label=label))


def get_pool_meta_data(pool: Pool, *, series_ids: Sequence[str] | None = None) -> str:
    """Pool metadata as JSON text (FR2)."""
    return dumps_payload(pool_meta_data(pool, series_ids=series_ids))
