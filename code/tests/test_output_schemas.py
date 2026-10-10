"""Schema validation of anomaly_verdicts.jsonl and decision_trace.jsonl (spec 02, FR5/FR6),
plus the verdict-line protocol that feeds the former."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.baseline_agent.ids import IdAliases
from src.baseline_agent.schemas import (
    AttributedUsage,
    ModelRequestRecord,
    RunEndRecord,
    RunStartRecord,
    SeriesRecord,
    ToolCallRecord,
    UsageFields,
    VerdictEvent,
    VerdictRecord,
    strip_timing,
    trace_adapter,
    validate_trace_file,
    validate_verdicts_file,
    zero_attributed,
)
from src.baseline_agent.verdicts import VerdictCollector, parse_verdicts
from src.baseline_agent.writers import VerdictWriter, dumps_line


def write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(dumps_line(r) + "\n" for r in rows), encoding="utf-8")
    return path


# --- FR5: verdict records ---------------------------------------------------------------


class TestVerdictRecord:
    def test_anomalous_requires_an_explanation(self):
        VerdictRecord(series_id="a", verdict="anomalous", explanation="spike at 12")
        with pytest.raises(ValidationError, match="explanation"):
            VerdictRecord(series_id="a", verdict="anomalous")
        with pytest.raises(ValidationError, match="explanation"):
            VerdictRecord(series_id="a", verdict="anomalous", explanation="   ")

    def test_normal_must_not_carry_an_explanation(self):
        rec = VerdictRecord(series_id="a", verdict="normal")
        assert rec.explanation is None
        with pytest.raises(ValidationError, match="normal verdict"):
            VerdictRecord(series_id="a", verdict="normal", explanation="looks fine")

    def test_verdict_vocabulary_and_extra_fields(self):
        with pytest.raises(ValidationError):
            VerdictRecord(series_id="a", verdict="suspicious", explanation="x")
        with pytest.raises(ValidationError):
            VerdictRecord(series_id="a", verdict="normal", score=0.9)
        with pytest.raises(ValidationError):
            VerdictRecord(series_id="", verdict="normal")
        with pytest.raises(ValidationError):
            VerdictRecord(series_id=7, verdict="normal")

    def test_file_validation(self, tmp_path):
        good = write_lines(
            tmp_path / "ok.jsonl",
            [
                {"series_id": "a", "verdict": "normal"},
                {"series_id": "b", "verdict": "anomalous", "explanation": "level shift from index 300"},
            ],
        )
        recs = validate_verdicts_file(good)
        assert [r.series_id for r in recs] == ["a", "b"]
        assert recs[0].explanation is None and recs[1].explanation

        dup = write_lines(tmp_path / "dup.jsonl", [{"series_id": "a", "verdict": "normal"}] * 2)
        with pytest.raises(ValueError, match="duplicate"):
            validate_verdicts_file(dup)
        bad = write_lines(tmp_path / "bad.jsonl", [{"series_id": "a", "verdict": "anomalous"}])
        with pytest.raises(ValueError, match="explanation"):
            validate_verdicts_file(bad)
        null_expl = write_lines(tmp_path / "null.jsonl", [{"series_id": "a", "verdict": "normal", "explanation": None}])
        assert validate_verdicts_file(null_expl)[0].explanation is None
        (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
        assert validate_verdicts_file(tmp_path / "empty.jsonl") == []

    def test_verdict_writer_keeps_one_record_per_series(self, tmp_path):
        w = VerdictWriter(tmp_path / "v.jsonl")
        assert w.put(VerdictRecord(series_id="a", verdict="normal")) is False
        assert w.put(VerdictRecord(series_id="b", verdict="anomalous", explanation="x")) is False
        assert w.put(VerdictRecord(series_id="a", verdict="anomalous", explanation="second look")) is True
        recs = validate_verdicts_file(tmp_path / "v.jsonl")
        assert [(r.series_id, r.verdict) for r in recs] == [("a", "anomalous"), ("b", "anomalous")]
        assert not (tmp_path / "v.jsonl.tmp").exists()
        lines = (tmp_path / "v.jsonl").read_text(encoding="utf-8").splitlines()
        assert '"explanation"' in lines[0] and "null" not in lines[0]


# --- the verdict-line protocol ----------------------------------------------------------


class TestVerdictParsing:
    def test_plain_lines(self):
        text = '{"series_id": "a", "verdict": "normal"}\n{"series_id": "b", "verdict": "anomalous", "explanation": "spike"}\nDONE 2'
        parsed = parse_verdicts(text)
        assert [(p.series_id, p.record.verdict) for p in parsed] == [("a", "normal"), ("b", "anomalous")]

    def test_fences_prose_pretty_json_and_arrays(self):
        text = (
            "Looking at the data.\n```json\n{\n  \"series_id\": \"a\",\n  \"verdict\": \"normal\"\n}\n```\n"
            "Batch: [{\"series_id\": \"b\", \"verdict\": \"anomalous\", \"explanation\": \"index 40 jumps to 9.0\"}, "
            "{\"series_id\": \"c\", \"verdict\": \"normal\"}]\nDONE 3"
        )
        parsed = parse_verdicts(text)
        assert [p.series_id for p in parsed] == ["a", "b", "c"]
        assert all(p.record is not None for p in parsed)

    def test_normalisation_and_notes(self):
        parsed = parse_verdicts('{"series_id": "a", "verdict": " Normal ", "explanation": "nothing odd", "confidence": 0.9}')
        (p,) = parsed
        assert p.record is not None and p.record.verdict == "normal" and p.record.explanation is None
        assert any("dropped" in n for n in p.notes) and any("extra field" in n for n in p.notes)

    def test_invalid_records_are_reported_not_raised(self):
        parsed = parse_verdicts('{"series_id": "a", "verdict": "anomalous"}\n{"series_id": "b", "verdict": "maybe"}')
        assert [p.record for p in parsed] == [None, None]
        assert "explanation" in parsed[0].error and "verdict must be one of" in parsed[1].error

    def test_malformed_line_is_flagged_once(self):
        parsed = parse_verdicts("{'series_id': 'a', 'verdict': 'normal'}\nDONE 1")
        assert len(parsed) == 1 and parsed[0].record is None and "malformed" in parsed[0].error
        assert parsed[0].series_id == "a"

    def test_prose_mentioning_series_id_is_not_a_verdict(self):
        assert parse_verdicts("All series_id values above were retrieved.\nDONE 3") == []
        assert parse_verdicts("I will now examine the next series.\nDONE 0") == []

    def test_decoded_objects_are_never_double_counted(self):
        # a non-string id is one invalid record, not an invalid record plus a malformed line
        parsed = parse_verdicts('{"series_id": 7, "verdict": "normal"}')
        assert len(parsed) == 1 and "non-empty string" in parsed[0].error
        # escaped characters inside a valid object do not make the raw line look broken
        parsed = parse_verdicts('{"series_id": "it\\"s", "verdict": "normal"}')
        assert len(parsed) == 1 and parsed[0].record is not None and parsed[0].series_id == 'it"s'

    def test_a_broken_line_is_not_hidden_by_a_similar_valid_one(self):
        text = '{"series_id": "s1", "verdict": "normal"}\n{"series_id": "s10", "verdict": normal}'
        parsed = parse_verdicts(text)
        assert [(p.series_id, p.record is not None) for p in parsed] == [("s1", True), ("s10", False)]

    def test_absurd_nesting_does_not_raise(self):
        parsed = parse_verdicts("[" * 5000 + '\n{"series_id": "a", "verdict": "normal"}')
        assert [p.series_id for p in parsed] == ["a"]

    def test_nested_object_inside_a_verdict_is_ignored(self):
        parsed = parse_verdicts('{"series_id": "a", "verdict": "normal", "extra": {"series_id": "b", "verdict": "normal"}}')
        assert [p.series_id for p in parsed] == ["a"]


class TestVerdictCollector:
    def test_episode_rules(self):
        c = VerdictCollector()
        known = {"a", "b", "c"}
        c.mark_examined("a")
        c.mark_examined("b")
        r = c.ingest('{"series_id": "a", "verdict": "normal"}\n{"series_id": "c", "verdict": "normal"}\n{"series_id": "zz", "verdict": "normal"}', 1, known)
        assert [x.series_id for x in r.accepted] == ["a"]
        assert [(cand.series_id, reason) for cand, reason in r.rejected] == [("c", "not_examined"), ("zz", "unknown_series_id")]
        assert c.missing_verdicts() == ["b"]
        # last verdict wins, revision counted, ingestion is idempotent per request
        r2 = c.ingest('{"series_id": "a", "verdict": "anomalous", "explanation": "index 3 = 99"}', 2, known)
        assert r2.superseded == ["a"] and c.revisions["a"] == 1 and c.final["a"].verdict == "anomalous"
        assert c.ingest("anything", 2, known).events == []
        assert c.results[1] is r and c.results[2] is r2
        assert len(c.records()) == 1 and c.n_rejected == 2

    def test_restating_a_verdict_is_not_a_revision(self):
        c = VerdictCollector()
        c.mark_examined("a")
        c.ingest('{"series_id": "a", "verdict": "normal"}', 1, {"a"})
        r = c.ingest('{"series_id": "a", "verdict": "normal"}\nDONE 1', 2, {"a"})
        assert r.restated == ["a"] and r.superseded == [] and c.revisions["a"] == 0
        assert r.events[0]["accepted"] and not r.events[0]["supersedes"]
        assert "restates the standing verdict" in r.events[0]["notes"]

    def test_whitespace_around_an_id_is_tolerated_and_noted(self):
        c = VerdictCollector()
        c.mark_examined("s1")
        r = c.ingest('{"series_id": " s1 ", "verdict": "normal"}', 1, {"s1"})
        assert [x.series_id for x in r.accepted] == ["s1"] and c.final["s1"].series_id == "s1"
        assert any("resolved from" in n for n in r.events[0]["notes"])

    def test_alias_resolution_and_explanation_chars(self):
        ids = IdAliases(["R1__YAHOO__real_1_full", "R1__NEK__x_full"])
        c = VerdictCollector()
        c.mark_examined("R1__NEK__x_full")
        text = '{"series_id": "s0002", "verdict": "anomalous", "explanation": "spike"}\n{"series_id": "R1__NEK__x_full", "verdict": "normal"}'
        r = c.ingest(text, 1, ids.served, resolve=ids.resolve)
        assert [x.series_id for x in r.accepted] == ["R1__NEK__x_full"]
        assert r.events[0]["series_id"] == "R1__NEK__x_full" and r.events[0]["explanation_chars"] == 5
        assert [(cand.series_id, reason) for cand, reason in r.rejected] == [("R1__NEK__x_full", "unknown_series_id")]

    def test_parser_failure_becomes_one_rejected_event(self, monkeypatch):
        import src.baseline_agent.verdicts as mod

        def boom(text):
            raise RuntimeError("no")

        monkeypatch.setattr(mod, "parse_verdicts", boom)
        c = VerdictCollector()
        r = c.ingest("x", 1, set())
        assert r.accepted == [] and r.events[0]["reason"] == "parse_error: RuntimeError"


# --- FR6: trace records -----------------------------------------------------------------


def minimal_trace() -> list[dict]:
    usage = UsageFields(input_tokens=100, output_tokens=10, total_tokens=110, cache_read_tokens=0, cache_write_tokens=0, details={}).model_dump()
    return [
        RunStartRecord(
            run_id="abc", config_fingerprint="cfg", pool={"name": "T1", "path": None, "n_series": 1, "fingerprint": "f"},
            order_seed=None, series_id_aliasing=False,
            model={"name": "scripted", "settings": {"temperature": 0.0}},
            usage_limits={"request_limit": 10, "tool_calls_limit": 10, "total_tokens_limit": None, "context_tokens_limit": None, "chars_per_token_estimate": 1.5, "output_allowance_tokens": 1024},
            retries={"tools": 1, "output": 2}, prompt={"system_prompt_sha256": "s", "user_prompt": "go"}, tools=["get_series", "get_pool_meta_data"],
            value_precision=None, harness={"pydantic_ai": "x"}, started_at="2026-01-01T00:00:00.000+00:00",
        ).model_dump(),
        ModelRequestRecord(
            request_index=1, is_retry=False, retry_kind=None, started_at="t", duration_ms=1.5, usage=usage, model_name="m", finish_reason="stop",
            history_chars=10, response_text_chars=0, n_tool_calls=1, requested_series=["a"], verdicts_accepted=0, verdicts_rejected=0,
        ).model_dump(),
        ToolCallRecord(
            call_index=1, request_index=1, tool_name="get_series", tool_call_id="id", args={"series_id": "a"}, series_id="a", ok=True,
            error=None, n_points=5, payload_chars=40, started_at="t", duration_ms=0.2,
        ).model_dump(),
        VerdictEvent(request_index=2, series_id="a", verdict="normal", explanation_chars=0, examined=True, accepted=True, reason=None, supersedes=False, notes=[]).model_dump(),
        SeriesRecord(
            series_id="a", n_points=5, examined=True, get_series_calls=1, failed_get_series_calls=0, first_request_index=1, last_request_index=2,
            tool_time_ms=0.2, verdict="normal", verdict_revisions=0, attributed_usage=AttributedUsage(requests=1.0, input_tokens=100.0, output_tokens=10.0, total_tokens=110.0),
        ).model_dump(),
        RunEndRecord(
            finished_at="t", duration_ms=3.0, termination={"reason": "completed"}, requests=2, retries={"tool": 0, "output": 0, "closing": 0},
            tool_calls={"get_series": 1, "get_pool_meta_data": 1, "failed": 0}, usage=usage, overhead_usage=zero_attributed(),
            series={"in_pool": 1, "examined": 1, "verdicted": 1, "anomalous": 0, "normal": 1, "examined_without_verdict": 0},
            verdicts={"accepted": 1, "rejected": 0, "revised": 0},
        ).model_dump(),
    ]


class TestTraceSchema:
    def test_minimal_trace_validates(self, tmp_path):
        path = write_lines(tmp_path / "trace.jsonl", minimal_trace())
        recs = validate_trace_file(path)
        assert [r.kind for r in recs] == ["run_start", "model_request", "tool_call", "verdict", "series", "run_end"]
        series = recs[4]
        assert isinstance(series.get_series_calls, int) and isinstance(series.tool_time_ms, float)
        assert isinstance(series.attributed_usage.total_tokens, float)
        assert isinstance(recs[1].usage.input_tokens, int) and recs[1].duration_ms == 1.5

    @pytest.mark.parametrize(
        "patch, message",
        [
            ((2, {"duration_ms": "fast"}), "duration_ms"),
            ((2, {"duration_ms": "1.5"}), "duration_ms"),
            ((4, {"get_series_calls": 1.5}), "get_series_calls"),
            ((4, {"get_series_calls": "3"}), "get_series_calls"),
            ((4, {"get_series_calls": True}), "get_series_calls"),
            ((4, {"attributed_usage": {"input_tokens": "many"}}), "attributed_usage"),
            ((4, {"attributed_usage": {}}), "attributed_usage"),
            ((5, {"termination": {"reason": "gave_up"}}), "termination"),
            ((1, {"usage": {"input_tokens": None}}), "usage"),
            ((1, {"usage": {}}), "usage"),
            ((5, {"tool_calls": {}}), "tool_calls"),
            ((2, {"ok": "yes"}), "ok"),
            ((3, {"kind": "judgement"}), "kind"),
            ((0, {"extra": 1}), "extra"),
        ],
    )
    def test_ill_typed_or_missing_fields_are_rejected(self, tmp_path, patch, message):
        rows = minimal_trace()
        idx, changes = patch
        rows[idx] = {**rows[idx], **changes}
        path = write_lines(tmp_path / "trace.jsonl", rows)
        with pytest.raises(ValueError, match=message):
            validate_trace_file(path)

    def test_missing_series_fields_are_rejected(self, tmp_path):
        rows = minimal_trace()
        del rows[4]["get_series_calls"]
        with pytest.raises(ValueError, match="get_series_calls"):
            validate_trace_file(write_lines(tmp_path / "t.jsonl", rows))

    def test_trace_must_be_closed_by_run_end(self, tmp_path):
        rows = minimal_trace()[:-1]
        with pytest.raises(ValueError, match="run_end"):
            validate_trace_file(write_lines(tmp_path / "open.jsonl", rows))
        with pytest.raises(ValueError, match="run_start"):
            validate_trace_file(write_lines(tmp_path / "headless.jsonl", minimal_trace()[1:]))

    def test_strip_timing_removes_only_timing(self):
        rows = strip_timing(minimal_trace())
        flat = json.dumps(rows)
        for key in ("started_at", "finished_at", "duration_ms", "tool_time_ms"):
            assert f'"{key}"' not in flat
        assert '"payload_chars"' in flat and '"input_tokens"' in flat

    def test_discriminator(self):
        rec = trace_adapter.validate_python({"kind": "verdict", "request_index": 1, "series_id": "a", "verdict": "normal", "explanation_chars": 0, "examined": True, "accepted": True, "reason": None, "supersedes": False, "notes": []})
        assert isinstance(rec, VerdictEvent)
        with pytest.raises(ValidationError):
            trace_adapter.validate_python({"kind": "nope"})
