"""Decision-trace recorder (spec 02, FR6 and the context-token NFR).

Records land in ``decision_trace.jsonl`` as they happen - a tool call the moment it
returns, a model request the moment its response is in - and are rolled up into
per-series records plus a ``run_end`` summary when the episode finishes, however it
finishes.

Token attribution
-----------------
Providers report usage per request, not per series, and one request can carry the payloads
of several series. The trace therefore charges each request to the series whose
``get_series`` payloads that request consumed - the successful calls executed right
before it - splitting the request's usage equally between them. A request that follows no
successful ``get_series`` (the prompt-only first turn, the turn after
``get_pool_meta_data``, retries, a closing message sent without a retrieval before it) is
booked as overhead. Shares are fractional so that they add up exactly to the run totals,
which ``test_runner.py`` checks.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from .pool import Pool
from .schemas import (
    AttributedUsage,
    ModelRequestRecord,
    RetryCounts,
    RetryKind,
    RunEndRecord,
    RunStartRecord,
    SeriesCounts,
    SeriesRecord,
    Termination,
    ToolCallCounts,
    ToolCallRecord,
    UsageFields,
    VerdictCounts,
    VerdictEvent,
    zero_attributed,
    zero_usage,
)
from .tools import is_error
from .verdicts import VerdictCollector
from .writers import JsonlWriter


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def usage_fields(usage: Any) -> UsageFields:
    """Convert a pydantic-ai ``RequestUsage`` / ``RunUsage`` (or a ``UsageFields``) into the trace's flat form."""
    details = {str(k): int(v) for k, v in (getattr(usage, "details", None) or {}).items()}
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return UsageFields(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_tokens=int(getattr(usage, "cache_read_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_write_tokens", 0) or 0),
        details=details,
    )


@dataclass
class ToolCallHandle:
    """What a tool wrapper fills in before its call returns."""

    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    series_id: str | None
    ok: bool = False
    error: str | None = "tool did not report a result"
    n_points: int | None = None
    payload_chars: int = 0

    def finish(self, text: str, *, n_points: int | None = None) -> None:
        self.payload_chars = len(text)
        if is_error(text):
            self.ok = False
            self.error = text
            self.n_points = None
        else:
            self.ok = True
            self.error = None
            self.n_points = n_points


@dataclass
class _SeriesStats:
    get_series_calls: int = 0
    failed_get_series_calls: int = 0
    tool_time_ms: float = 0.0
    first_request_index: int | None = None
    last_request_index: int | None = None
    served_requests: list[int] = field(default_factory=list)


class TraceRecorder:
    """Owns ``decision_trace.jsonl`` for one episode."""

    def __init__(self, writer: JsonlWriter, pool: Pool):
        self.writer = writer
        self.pool = pool
        self.tool_calls: list[ToolCallRecord] = []
        self.requests: list[ModelRequestRecord] = []
        self.verdict_events: list[VerdictEvent] = []
        self.series: OrderedDict[str, _SeriesStats] = OrderedDict()
        self.retries: dict[str, int] = {"tool": 0, "output": 0, "closing": 0}
        self._t0 = time.perf_counter()
        self._started_at = now_iso()

    # --- lifecycle --------------------------------------------------------------------

    def start(self, record: RunStartRecord) -> None:
        self.writer.write(record)

    @contextmanager
    def tool_call(
        self,
        tool_name: str,
        tool_call_id: str | None,
        args: dict[str, Any],
        *,
        request_index: int,
        series_id: str | None = None,
    ) -> Iterator[ToolCallHandle]:
        """Time one tool invocation and write its record when the block exits."""
        handle = ToolCallHandle(tool_name, tool_call_id or "", dict(args), series_id)
        started_at = now_iso()
        t0 = time.perf_counter()
        try:
            yield handle
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            self._record_call(handle, request_index, started_at, duration_ms)

    def invalid_tool_call(self, tool_name: str, tool_call_id: str | None, args: Any, error: str, *, request_index: int) -> None:
        """A call the harness refused to execute (bad arguments or unknown tool)."""
        raw_args = args if isinstance(args, dict) else {"raw": args}
        sid = raw_args.get("series_id")
        handle = ToolCallHandle(tool_name, tool_call_id or "", raw_args, sid if isinstance(sid, str) else None)
        handle.error = error
        self._record_call(handle, request_index, now_iso(), 0.0)

    def skipped_tool_call(self, tool_name: str, tool_call_id: str | None, args: Any, *, request_index: int, series_id: str | None) -> None:
        """A call the model issued but the budget stop left unexecuted."""
        raw_args = args if isinstance(args, dict) else {"raw": args}
        handle = ToolCallHandle(tool_name, tool_call_id or "", raw_args, series_id)
        handle.error = "not executed: the episode's token budget was exhausted before this call"
        self._record_call(handle, request_index, now_iso(), 0.0)

    def _record_call(self, handle: ToolCallHandle, request_index: int, started_at: str, duration_ms: float) -> None:
        record = ToolCallRecord(
            call_index=len(self.tool_calls) + 1,
            request_index=request_index,
            tool_name=handle.tool_name,
            tool_call_id=handle.tool_call_id,
            args=handle.args,
            series_id=handle.series_id,
            ok=handle.ok,
            error=handle.error,
            n_points=handle.n_points,
            payload_chars=handle.payload_chars,
            started_at=started_at,
            duration_ms=round(duration_ms, 3),
        )
        self.tool_calls.append(record)
        self.writer.write(record)
        if handle.tool_name == "get_series" and handle.series_id is not None and handle.series_id in self.pool:
            stats = self.series.setdefault(handle.series_id, _SeriesStats())
            stats.get_series_calls += 1
            if not handle.ok:
                stats.failed_get_series_calls += 1
            stats.tool_time_ms += duration_ms
            self._touch(stats, request_index)
            if handle.ok:
                stats.served_requests.append(request_index + 1)

    def model_request(
        self,
        *,
        request_index: int,
        retry_kind: RetryKind | None,
        started_at: str,
        duration_ms: float,
        usage: Any,
        model_name: str | None,
        finish_reason: str | None,
        history_chars: int,
        response_text_chars: int,
        n_tool_calls: int,
        requested_series: Sequence[str],
        verdicts_accepted: int,
        verdicts_rejected: int,
    ) -> ModelRequestRecord:
        record = ModelRequestRecord(
            request_index=request_index,
            is_retry=retry_kind is not None,
            retry_kind=retry_kind,
            started_at=started_at,
            duration_ms=round(duration_ms, 3),
            usage=usage_fields(usage),
            model_name=model_name,
            finish_reason=finish_reason,
            history_chars=history_chars,
            response_text_chars=response_text_chars,
            n_tool_calls=n_tool_calls,
            requested_series=list(requested_series),
            verdicts_accepted=verdicts_accepted,
            verdicts_rejected=verdicts_rejected,
        )
        if retry_kind is not None:
            self.retries[retry_kind] += 1
        self.requests.append(record)
        self.writer.write(record)
        return record

    def verdicts(self, events: Sequence[dict[str, Any]]) -> None:
        for ev in events:
            record = VerdictEvent(**ev)
            self.verdict_events.append(record)
            self.writer.write(record)
            if record.series_id in self.pool:
                stats = self.series.setdefault(record.series_id, _SeriesStats())
                self._touch(stats, record.request_index)

    def total_tokens(self) -> int:
        return sum(r.usage.total_tokens for r in self.requests)

    def finish(self, termination: Termination, collector: VerdictCollector) -> RunEndRecord:
        """Write the per-series roll-ups and the ``run_end`` record; returns the latter."""
        attributed, overhead = self._attribute_usage()
        for sid, stats in self.series.items():
            verdict = collector.final.get(sid)
            self.writer.write(
                SeriesRecord(
                    series_id=sid,
                    n_points=self.pool.length_of(sid) if sid in self.pool else None,
                    examined=sid in collector.examined,
                    get_series_calls=stats.get_series_calls,
                    failed_get_series_calls=stats.failed_get_series_calls,
                    first_request_index=stats.first_request_index,
                    last_request_index=stats.last_request_index,
                    tool_time_ms=round(stats.tool_time_ms, 3),
                    verdict=verdict.verdict if verdict else None,
                    verdict_revisions=collector.revisions.get(sid, 0),
                    attributed_usage=attributed.get(sid, zero_attributed()),
                )
            )
        finals = collector.records()
        n_anomalous = sum(r.verdict == "anomalous" for r in finals)
        totals = zero_usage()
        for r in self.requests:
            totals = UsageFields(
                input_tokens=totals.input_tokens + r.usage.input_tokens,
                output_tokens=totals.output_tokens + r.usage.output_tokens,
                total_tokens=totals.total_tokens + r.usage.total_tokens,
                cache_read_tokens=totals.cache_read_tokens + r.usage.cache_read_tokens,
                cache_write_tokens=totals.cache_write_tokens + r.usage.cache_write_tokens,
                details=_merge_details(totals.details, r.usage.details),
            )
        record = RunEndRecord(
            finished_at=now_iso(),
            duration_ms=round((time.perf_counter() - self._t0) * 1000.0, 3),
            termination=termination,
            requests=len(self.requests),
            retries=RetryCounts(tool=self.retries["tool"], output=self.retries["output"], closing=self.retries["closing"]),
            tool_calls=ToolCallCounts(
                get_series=sum(c.tool_name == "get_series" for c in self.tool_calls),
                get_pool_meta_data=sum(c.tool_name == "get_pool_meta_data" for c in self.tool_calls),
                failed=sum(not c.ok for c in self.tool_calls),
            ),
            usage=totals,
            overhead_usage=overhead,
            series=SeriesCounts(
                in_pool=len(self.pool),
                examined=len(collector.examined),
                verdicted=len(finals),
                anomalous=n_anomalous,
                normal=len(finals) - n_anomalous,
                examined_without_verdict=len(collector.missing_verdicts()),
            ),
            verdicts=VerdictCounts(
                accepted=sum(e.accepted for e in self.verdict_events),
                rejected=sum(not e.accepted for e in self.verdict_events),
                revised=sum(collector.revisions.values()),
            ),
        )
        self.writer.write(record)
        return record

    # --- helpers ----------------------------------------------------------------------

    @staticmethod
    def _touch(stats: _SeriesStats, request_index: int) -> None:
        if stats.first_request_index is None or request_index < stats.first_request_index:
            stats.first_request_index = request_index
        if stats.last_request_index is None or request_index > stats.last_request_index:
            stats.last_request_index = request_index

    def _attribute_usage(self) -> tuple[dict[str, AttributedUsage], AttributedUsage]:
        served: dict[int, list[str]] = {}
        for sid, stats in self.series.items():
            for k in stats.served_requests:
                served.setdefault(k, []).append(sid)
        shares: dict[str, list[float]] = {}
        overhead = [0.0, 0.0, 0.0, 0.0]
        for r in self.requests:
            targets = served.get(r.request_index, [])
            if not targets:
                overhead[0] += 1.0
                overhead[1] += r.usage.input_tokens
                overhead[2] += r.usage.output_tokens
                overhead[3] += r.usage.total_tokens
                continue
            w = 1.0 / len(targets)
            for sid in targets:
                acc = shares.setdefault(sid, [0.0, 0.0, 0.0, 0.0])
                acc[0] += w
                acc[1] += w * r.usage.input_tokens
                acc[2] += w * r.usage.output_tokens
                acc[3] += w * r.usage.total_tokens
        attributed = {
            sid: AttributedUsage(requests=round(a[0], 6), input_tokens=round(a[1], 6), output_tokens=round(a[2], 6), total_tokens=round(a[3], 6))
            for sid, a in shares.items()
        }
        return attributed, AttributedUsage(
            requests=round(overhead[0], 6),
            input_tokens=round(overhead[1], 6),
            output_tokens=round(overhead[2], 6),
            total_tokens=round(overhead[3], 6),
        )


def _merge_details(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out
