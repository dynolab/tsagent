"""Verdict protocol: how verdicts travel from the agent's text to ``anomaly_verdicts.jsonl``.

The agent has exactly two tools and no output tool (spec 02, FR3), so verdicts are carried
by its text: one JSON object per verdict, ``{"series_id": ..., "verdict": ...,
"explanation": ...}``. The parser here is deliberately lenient about *where* such objects
appear (bare lines, code fences, pretty-printed objects, arrays, surrounding prose) and
strict about *what* they must contain, which is exactly :class:`VerdictRecord`.

:class:`VerdictCollector` applies the episode rules on top of parsing: a verdict counts
only for a series the agent actually retrieved, the last verdict for a series wins, a
restated verdict changes nothing, and everything - accepted or rejected - is reported
back so the trace can record it.
"""

from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Collection

from pydantic import ValidationError

from .schemas import VERDICT_VALUES, VerdictRecord

_DECODER = json.JSONDecoder()
_VERDICT_KEYS = frozenset({"series_id", "verdict", "explanation"})
_ATTEMPT = re.compile(r"series_id\W{0,3}[:=]")  # a line that tried to be a verdict object
_ID_AFTER_KEY = re.compile(r"""series_id\W*["']([^"']+)["']""")


@dataclass(frozen=True)
class ParsedVerdict:
    """One verdict candidate found in the text."""

    raw: dict[str, Any]
    record: VerdictRecord | None
    error: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def series_id(self) -> str:
        sid = self.raw.get("series_id")
        return sid if isinstance(sid, str) else str(sid)


def extract_json_values(text: str) -> list[tuple[Any, int, int]]:
    """Every JSON object or array embedded in ``text``, with its character span.

    Scans for ``{`` / ``[`` and attempts a decode from each; a successful decode consumes
    its span, so nested structures are returned once, at the outermost level. Absurdly
    deep nesting (a model stuck in a ``[[[[`` loop) is skipped like any other non-JSON.
    """
    values: list[tuple[Any, int, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos] not in "{[":
            pos += 1
            continue
        try:
            value, end = _DECODER.raw_decode(text, pos)
        except (json.JSONDecodeError, RecursionError):
            pos += 1
            continue
        values.append((value, pos, end))
        pos = end
    return values


def _candidates(value: Any, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 32:
        return []
    out: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "series_id" in value:
            out.append(value)
        else:
            for v in value.values():
                out.extend(_candidates(v, depth + 1))
    elif isinstance(value, list):
        for v in value:
            out.extend(_candidates(v, depth + 1))
    return out


def _normalise(raw: dict[str, Any]) -> tuple[VerdictRecord | None, str | None, tuple[str, ...]]:
    notes: list[str] = []
    sid = raw.get("series_id")
    if not isinstance(sid, str) or not sid.strip():
        return None, "series_id must be a non-empty string", ()
    verdict = raw.get("verdict")
    if isinstance(verdict, str):
        cleaned = verdict.strip().lower()
        if cleaned != verdict:
            notes.append("verdict value normalised")
        verdict = cleaned
    if verdict not in VERDICT_VALUES:
        return None, f"verdict must be one of {list(VERDICT_VALUES)}, got {verdict!r}", tuple(notes)
    explanation = raw.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        explanation = json.dumps(explanation, ensure_ascii=False)
        notes.append("explanation coerced to text")
    if verdict == "normal" and explanation is not None:
        explanation = None
        notes.append("explanation dropped for a normal verdict")
    extra = sorted(set(raw) - _VERDICT_KEYS)
    if extra:
        notes.append(f"ignored extra field(s) {extra}")
    try:
        record = VerdictRecord(series_id=sid, verdict=verdict, explanation=explanation)
    except ValidationError as exc:
        return None, exc.errors()[0]["msg"], tuple(notes)
    return record, None, tuple(notes)


def parse_verdicts(text: str) -> list[ParsedVerdict]:
    """All verdict candidates in ``text``.

    Candidates are JSON objects carrying a ``series_id`` key (anywhere in the text, at any
    nesting depth), plus any line outside every decoded value that still looks like an
    attempt at one (``series_id`` followed by a colon) - a broken object, reported so the
    trace can count it.
    """
    parsed: list[ParsedVerdict] = []
    spans: list[tuple[int, int]] = []
    for value, start, end in extract_json_values(text):
        spans.append((start, end))
        for raw in _candidates(value):
            record, error, notes = _normalise(raw)
            parsed.append(ParsedVerdict(raw=raw, record=record, error=error, notes=notes))
    offset = 0
    for line in text.splitlines(keepends=True):
        start, end = offset, offset + len(line)
        offset = end
        if not _ATTEMPT.search(line):
            continue
        if any(a < end and start < b for a, b in spans):
            continue
        parsed.append(
            ParsedVerdict(
                raw={"series_id": _guess_id(line), "line": line.strip()},
                record=None,
                error="malformed verdict line (not valid JSON)",
            )
        )
    return parsed


def _guess_id(line: str) -> str:
    """Best-effort id for the trace: the first quoted token following ``series_id``."""
    match = _ID_AFTER_KEY.search(line)
    if match:
        return match.group(1).strip()
    tail = line.split("series_id", 1)[1]
    return tail.strip(" :=,\"'{}")[:80] or "?"


@dataclass
class IngestResult:
    accepted: list[VerdictRecord] = field(default_factory=list)
    rejected: list[tuple[ParsedVerdict, str]] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    restated: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


Resolver = Callable[[str], "str | None"]


class VerdictCollector:
    """Episode-level verdict state: which series were examined, which verdict stands.

    Ids are resolved through ``resolve`` (the id view of the episode: plain or aliased);
    without one, an exact match against ``known_ids`` and then a whitespace-trimmed one.
    """

    def __init__(self) -> None:
        self.examined: OrderedDict[str, int] = OrderedDict()
        self.final: OrderedDict[str, VerdictRecord] = OrderedDict()
        self.revisions: Counter[str] = Counter()
        self.n_rejected = 0
        self.results: dict[int, IngestResult] = {}
        self._ingested_requests: set[int] = set()

    def mark_examined(self, series_id: str) -> None:
        self.examined[series_id] = self.examined.get(series_id, 0) + 1

    def has_ingested(self, request_index: int) -> bool:
        return request_index in self._ingested_requests

    def missing_verdicts(self) -> list[str]:
        """Examined series without a standing verdict, in examination order."""
        return [sid for sid in self.examined if sid not in self.final]

    def ingest(self, text: str, request_index: int, known_ids: Collection[str], *, resolve: Resolver | None = None) -> IngestResult:
        """Parse ``text`` and apply the episode rules. Idempotent per ``request_index``."""
        result = IngestResult()
        if request_index in self._ingested_requests:
            return result
        self._ingested_requests.add(request_index)
        self.results[request_index] = result
        if not text:
            return result
        resolver = resolve or _plain_resolver(known_ids)
        try:
            candidates = parse_verdicts(text)
        except Exception as exc:  # a degenerate reply must not end the episode
            self.n_rejected += 1
            result.rejected.append((ParsedVerdict(raw={"series_id": "?"}, record=None, error=type(exc).__name__), f"parse_error: {type(exc).__name__}"))
            result.events.append(_event(request_index, "?", "?", 0, False, False, f"parse_error: {type(exc).__name__}", False, []))
            return result
        for cand in candidates:
            raw_sid = cand.series_id
            real = resolver(raw_sid)
            notes = list(cand.notes)
            if real is not None and real != raw_sid:
                notes.append(f"series_id resolved from {raw_sid!r}")
            sid = real if real is not None else raw_sid
            examined = sid in self.examined
            reason: str | None = None
            if cand.record is None:
                reason = f"invalid_record: {cand.error}"
            elif real is None:
                reason = "unknown_series_id"
            elif not examined:
                reason = "not_examined"
            supersedes = False
            explanation_chars = 0
            if reason is None:
                assert cand.record is not None
                record = VerdictRecord(series_id=sid, verdict=cand.record.verdict, explanation=cand.record.explanation)
                explanation_chars = len(record.explanation or "")
                standing = self.final.get(sid)
                if standing is not None and standing == record:
                    notes.append("restates the standing verdict")
                    result.restated.append(sid)
                elif standing is not None:
                    supersedes = True
                    self.revisions[sid] += 1
                    result.superseded.append(sid)
                self.final[sid] = record
                result.accepted.append(record)
            else:
                self.n_rejected += 1
                result.rejected.append((cand, reason))
                explanation = cand.raw.get("explanation")
                explanation_chars = len(explanation) if isinstance(explanation, str) else 0
            result.events.append(
                _event(request_index, sid, str(cand.raw.get("verdict")), explanation_chars, examined, reason is None, reason, supersedes, notes)
            )
        return result

    def records(self) -> list[VerdictRecord]:
        return list(self.final.values())


def _plain_resolver(known_ids: Collection[str]) -> Resolver:
    def resolve(model_id: str) -> str | None:
        if model_id in known_ids:
            return model_id
        trimmed = model_id.strip()
        return trimmed if trimmed in known_ids else None

    return resolve


def _event(request_index: int, series_id: str, verdict: str, explanation_chars: int, examined: bool, accepted: bool, reason: str | None, supersedes: bool, notes: list[str]) -> dict[str, Any]:
    return {
        "request_index": request_index,
        "series_id": series_id,
        "verdict": verdict,
        "explanation_chars": explanation_chars,
        "examined": examined,
        "accepted": accepted,
        "reason": reason,
        "supersedes": supersedes,
        "notes": list(notes),
    }
