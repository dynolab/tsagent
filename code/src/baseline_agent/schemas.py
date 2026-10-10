"""Output contracts of a baseline-agent episode (spec 02, FR5 and FR6).

Two files are written per episode:

``anomaly_verdicts.jsonl``
    One :class:`VerdictRecord` per series the agent examined - its final verdict.

``decision_trace.jsonl``
    A chronological log of :data:`TraceRecord` objects discriminated by ``kind``:
    ``run_start`` (configuration), ``model_request`` (one per LLM call: usage and timing),
    ``tool_call`` (one per tool invocation: arguments, outcome, timing), ``verdict`` (every
    verdict as it was parsed, accepted or not), ``series`` (per-series roll-up: call count,
    timing, attributed token usage) and ``run_end`` (totals and how the episode ended).

The models validate strictly - ``"3"`` is not ``3`` and ``1`` is not ``True`` - and carry no
defaults for the counted fields, so a file that passes :func:`validate_trace_file` really
does hold every field the acceptance criteria name, with the right type.

Timing fields hold wall-clock measurements and therefore differ between otherwise
identical runs; :data:`TIMING_FIELDS` names them so a determinism check can skip them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

Verdict = Literal["anomalous", "normal"]
VERDICT_VALUES: tuple[str, ...] = ("anomalous", "normal")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


# --- anomaly_verdicts.jsonl -------------------------------------------------------------


class VerdictRecord(_Strict):
    """One line of ``anomaly_verdicts.jsonl``.

    ``explanation`` is present exactly when ``verdict == "anomalous"``.
    """

    series_id: str = Field(min_length=1)
    verdict: Verdict
    explanation: str | None = None

    @model_validator(mode="after")
    def _explanation_iff_anomalous(self) -> "VerdictRecord":
        if self.verdict == "anomalous":
            if self.explanation is None or not self.explanation.strip():
                raise ValueError("an anomalous verdict needs a non-empty explanation")
        elif self.explanation is not None:
            raise ValueError("a normal verdict must not carry an explanation")
        return self


# --- decision_trace.jsonl ---------------------------------------------------------------


class UsageFields(_Strict):
    """Token usage as reported by the provider (integers, exact)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    details: dict[str, int]


def zero_usage() -> UsageFields:
    return UsageFields(input_tokens=0, output_tokens=0, total_tokens=0, cache_read_tokens=0, cache_write_tokens=0, details={})


class AttributedUsage(_Strict):
    """A share of the run's usage charged to one series.

    Fractional on purpose: a request that served several series at once is split equally
    between them, so the per-series shares plus the overhead add up to the run totals.
    """

    requests: float
    input_tokens: float
    output_tokens: float
    total_tokens: float


def zero_attributed() -> AttributedUsage:
    return AttributedUsage(requests=0.0, input_tokens=0.0, output_tokens=0.0, total_tokens=0.0)


class PoolInfo(_Strict):
    name: str
    path: str | None
    n_series: int
    fingerprint: str


class ModelInfo(_Strict):
    name: str
    settings: dict[str, Any]
    provider_base_url: str | None = None


class LimitsInfo(_Strict):
    """The episode caps: pydantic-ai's (checked after the fact) and the harness's projection."""

    request_limit: int | None
    tool_calls_limit: int | None
    total_tokens_limit: int | None
    context_tokens_limit: int | None
    chars_per_token_estimate: float
    output_allowance_tokens: int


class PromptInfo(_Strict):
    system_prompt_sha256: str
    user_prompt: str


class RetriesInfo(_Strict):
    tools: int
    output: int


class RunStartRecord(_Strict):
    kind: Literal["run_start"] = "run_start"
    run_id: str
    config_fingerprint: str
    pool: PoolInfo
    order_seed: int | None
    series_id_aliasing: bool
    model: ModelInfo
    usage_limits: LimitsInfo
    retries: RetriesInfo
    prompt: PromptInfo
    tools: list[str]
    value_precision: int | None
    harness: dict[str, str]
    started_at: str


class ToolCallRecord(_Strict):
    """One tool invocation, in execution order. ``args`` is what the model sent; ``series_id``
    is the pool id it resolved to (or the raw string when it did not resolve)."""

    kind: Literal["tool_call"] = "tool_call"
    call_index: int
    request_index: int
    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    series_id: str | None
    ok: bool
    error: str | None
    n_points: int | None
    payload_chars: int
    started_at: str
    duration_ms: float


RetryKind = Literal["tool", "output", "closing"]
"""Why a request carries a harness prompt: invalid tool arguments, verdicts missing when
the model stopped, or the closing turn after a budget stop."""


class ModelRequestRecord(_Strict):
    """One LLM request/response round trip."""

    kind: Literal["model_request"] = "model_request"
    request_index: int
    is_retry: bool
    retry_kind: RetryKind | None
    started_at: str
    duration_ms: float
    usage: UsageFields
    model_name: str | None
    finish_reason: str | None
    history_chars: int
    response_text_chars: int
    n_tool_calls: int
    requested_series: list[str]
    verdicts_accepted: int
    verdicts_rejected: int


class VerdictEvent(_Strict):
    """A verdict as it was parsed from the agent's text, whether or not it was kept.

    The explanation text itself is not repeated here: it lives in ``anomaly_verdicts.jsonl``.
    """

    kind: Literal["verdict"] = "verdict"
    request_index: int
    series_id: str
    verdict: str
    explanation_chars: int
    examined: bool
    accepted: bool
    reason: str | None
    supersedes: bool
    notes: list[str]


class SeriesRecord(_Strict):
    """Per-series roll-up for every series the agent touched."""

    kind: Literal["series"] = "series"
    series_id: str
    n_points: int | None
    examined: bool
    get_series_calls: int
    failed_get_series_calls: int
    first_request_index: int | None
    last_request_index: int | None
    tool_time_ms: float
    verdict: Verdict | None
    verdict_revisions: int
    attributed_usage: AttributedUsage


TerminationReason = Literal["completed", "budget", "usage_limit", "model_error", "protocol_error", "interrupted"]


class Termination(_Strict):
    reason: TerminationReason
    detail: str | None = None


class ToolCallCounts(_Strict):
    get_series: int
    get_pool_meta_data: int
    failed: int


class SeriesCounts(_Strict):
    in_pool: int
    examined: int
    verdicted: int
    anomalous: int
    normal: int
    examined_without_verdict: int


class VerdictCounts(_Strict):
    accepted: int
    rejected: int
    revised: int


class RetryCounts(_Strict):
    tool: int
    output: int
    closing: int


class RunEndRecord(_Strict):
    kind: Literal["run_end"] = "run_end"
    finished_at: str
    duration_ms: float
    termination: Termination
    requests: int
    retries: RetryCounts
    tool_calls: ToolCallCounts
    usage: UsageFields
    overhead_usage: AttributedUsage
    series: SeriesCounts
    verdicts: VerdictCounts


TraceRecord = Annotated[
    Union[RunStartRecord, ToolCallRecord, ModelRequestRecord, VerdictEvent, SeriesRecord, RunEndRecord],
    Field(discriminator="kind"),
]

trace_adapter: TypeAdapter[Any] = TypeAdapter(TraceRecord)
verdict_adapter: TypeAdapter[VerdictRecord] = TypeAdapter(VerdictRecord)

TIMING_FIELDS: frozenset[str] = frozenset({"started_at", "finished_at", "duration_ms", "tool_time_ms"})


# --- file-level validation --------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Parse a ``.jsonl`` file; blank lines are skipped, anything else must be a JSON object."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from None
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def validate_verdicts_file(path: str | Path) -> list[VerdictRecord]:
    """Validate ``anomaly_verdicts.jsonl``: every record well-formed, one record per series."""
    records: list[VerdictRecord] = []
    seen: set[str] = set()
    for i, row in enumerate(read_jsonl(path), start=1):
        try:
            rec = verdict_adapter.validate_python(row)
        except ValidationError as exc:
            raise ValueError(f"{path}:{i}: {exc.errors()[0]['msg']}") from None
        if rec.series_id in seen:
            raise ValueError(f"{path}:{i}: duplicate verdict for series {rec.series_id!r}")
        seen.add(rec.series_id)
        records.append(rec)
    return records


def validate_trace_file(path: str | Path) -> list[Any]:
    """Validate ``decision_trace.jsonl``: typed records, opened by ``run_start``, closed by ``run_end``."""
    records: list[Any] = []
    for i, row in enumerate(read_jsonl(path), start=1):
        try:
            records.append(trace_adapter.validate_python(row))
        except ValidationError as exc:
            err = exc.errors()[0]
            loc = ".".join(str(p) for p in err["loc"])
            raise ValueError(f"{path}:{i}: {loc}: {err['msg']}") from None
    if not records:
        raise ValueError(f"{path}: empty trace")
    if not isinstance(records[0], RunStartRecord):
        raise ValueError(f"{path}: first record must be run_start, got {records[0].kind!r}")
    if not isinstance(records[-1], RunEndRecord):
        raise ValueError(f"{path}: last record must be run_end, got {records[-1].kind!r} (run did not finish cleanly)")
    starts = sum(isinstance(r, RunStartRecord) for r in records)
    ends = sum(isinstance(r, RunEndRecord) for r in records)
    if starts != 1 or ends != 1:
        raise ValueError(f"{path}: expected one run_start and one run_end, found {starts} and {ends}")
    return records


def strip_timing(obj: Any) -> Any:
    """Recursively drop :data:`TIMING_FIELDS` so two replays can be compared field-for-field."""
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if k not in TIMING_FIELDS}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj
