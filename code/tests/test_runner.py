"""End-to-end episodes on the tiny pool with the scripted stand-in model: outputs, trace
consistency, determinism (FR7), graceful termination and the protocol counter-measures."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai import messages as M
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from src.baseline_agent.prompt import SYSTEM_PROMPT, USER_PROMPT, prompt_fingerprint
from src.baseline_agent.runner import (
    ALIASES_FILE,
    MESSAGES_FILE,
    SUMMARY_FILE,
    TRACE_FILE,
    VERDICTS_FILE,
    EpisodeConfig,
    compare_runs,
    history_chars,
    run_episode,
)
from src.baseline_agent.schemas import (
    ModelRequestRecord,
    RunEndRecord,
    RunStartRecord,
    SeriesRecord,
    ToolCallRecord,
    VerdictEvent,
    validate_trace_file,
    validate_verdicts_file,
)
from src.baseline_agent.scripted import ScriptedPolicy
from src.baseline_agent.tools import get_series


def config(**overrides) -> EpisodeConfig:
    base = dict(model="scripted", total_tokens_limit=None)
    base.update(overrides)
    return EpisodeConfig(**base)


def run(pool, tmp_path: Path, name: str = "run", **overrides):
    return run_episode(pool, config(**overrides), tmp_path / name)


def trace_of(result):
    return validate_trace_file(result.files[TRACE_FILE])


def by_kind(records, cls):
    return [r for r in records if isinstance(r, cls)]


def messages_of(result) -> list[dict]:
    return json.loads(result.files[MESSAGES_FILE].read_text(encoding="utf-8"))


def usage(n: int = 1) -> RequestUsage:
    return RequestUsage(input_tokens=n, output_tokens=1)


class TestHappyPath:
    def test_outputs_exist_and_validate(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path)
        for name in (VERDICTS_FILE, TRACE_FILE, MESSAGES_FILE, SUMMARY_FILE):
            assert result.files[name].is_file()
        assert ALIASES_FILE in result.files  # aliasing is on by default
        assert ALIASES_FILE not in run(tiny_pool, tmp_path, "plain", alias_series_ids=False).files
        verdicts = validate_verdicts_file(result.files[VERDICTS_FILE])
        records = trace_of(result)
        end = records[-1]
        assert result.termination.reason == "completed" and end.termination.reason == "completed"
        assert [v.series_id for v in verdicts] == tiny_pool.series_ids  # examination order == served order
        assert {v.series_id: v.verdict for v in verdicts}[tiny_pool.series_ids[2]] == "anomalous"
        assert end.series.model_dump() == {"in_pool": 5, "examined": 5, "verdicted": 5, "anomalous": 2, "normal": 3, "examined_without_verdict": 0}
        summary = json.loads(result.files[SUMMARY_FILE].read_text(encoding="utf-8"))
        assert summary["run_id"] == result.run_id and summary["series"]["verdicted"] == 5

    def test_trace_is_internally_consistent(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path)
        records = trace_of(result)
        end: RunEndRecord = records[-1]
        requests = by_kind(records, ModelRequestRecord)
        calls = by_kind(records, ToolCallRecord)
        series = by_kind(records, SeriesRecord)
        n_responses = sum(m["kind"] == "response" for m in messages_of(result))
        assert len(requests) == end.requests == n_responses == 7
        assert [r.request_index for r in requests] == list(range(1, 8))
        assert end.tool_calls.model_dump() == {"get_series": 5, "get_pool_meta_data": 1, "failed": 0}
        assert len(calls) == 6 and all(c.ok for c in calls)
        assert [c.request_index for c in calls] == [1, 2, 3, 4, 5, 6]
        assert len(series) == 5 and all(s.get_series_calls == 1 and s.examined and s.verdict for s in series)
        assert all(s.first_request_index is not None and s.first_request_index <= s.last_request_index for s in series)
        # usage: per-request records add up to the totals, and attribution conserves them
        assert sum(r.usage.total_tokens for r in requests) == end.usage.total_tokens > 0
        assert sum(r.usage.input_tokens for r in requests) == end.usage.input_tokens
        attributed = sum(s.attributed_usage.total_tokens for s in series) + end.overhead_usage.total_tokens
        assert attributed == pytest.approx(end.usage.total_tokens)
        assert sum(s.attributed_usage.requests for s in series) + end.overhead_usage.requests == pytest.approx(end.requests)
        assert end.overhead_usage.requests == 2  # the prompt-only first turn and the metadata turn
        assert sum(r.verdicts_accepted for r in requests) == end.verdicts.accepted == 5
        assert by_kind(records, VerdictEvent)[0].request_index == 3
        assert end.retries.model_dump() == {"tool": 0, "output": 0, "closing": 0}
        # the trace is chronological: a request's verdict events follow its own record
        kinds = [r.kind for r in records]
        assert kinds[:4] == ["run_start", "model_request", "tool_call", "model_request"]
        assert kinds[kinds.index("verdict") - 2:kinds.index("verdict") + 1] == ["tool_call", "model_request", "verdict"]

    def test_history_chars_counts_the_prompt_once(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path)
        records = trace_of(result)
        requests = by_kind(records, ModelRequestRecord)
        calls = by_kind(records, ToolCallRecord)
        assert requests[0].history_chars == len(SYSTEM_PROMPT.strip()) + len(USER_PROMPT)
        for prev, nxt in zip(requests, requests[1:]):
            returned = sum(c.payload_chars for c in calls if c.request_index == prev.request_index)
            growth = nxt.history_chars - prev.history_chars
            assert returned + prev.response_text_chars <= growth < returned + prev.response_text_chars + 500
        # and the function agrees with a direct count of the recorded conversation
        messages = M.ModelMessagesTypeAdapter.validate_python(messages_of(result))
        assert history_chars(messages, instructions=SYSTEM_PROMPT) == history_chars(messages)

    def test_replay_is_deterministic(self, tiny_pool, tmp_path):
        a = run(tiny_pool, tmp_path, "a")
        b = run(tiny_pool, tmp_path, "b")
        assert a.run_id == b.run_id
        diff = compare_runs(a.output_dir, b.output_dir)
        assert diff.identical, diff.differences
        assert a.files[VERDICTS_FILE].read_bytes() == b.files[VERDICTS_FILE].read_bytes()

    def test_batched_tool_calls_run_in_order_and_replay_identically(self, tiny_pool, tmp_path):
        ids = tiny_pool.series_ids

        def batching(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_pool_meta_data", {})], usage=usage())
            if any(isinstance(p, M.ToolReturnPart) and p.tool_name == "get_pool_meta_data" for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": s}) for s in ids], usage=usage())
            lines = [json.dumps({"series_id": s, "verdict": "normal"}) for s in ids]
            return M.ModelResponse(parts=[M.TextPart("\n".join(lines + ["DONE 5"]))], usage=usage())

        # The policy addresses series by pool id, so it runs with aliasing off.
        results = [run_episode(tiny_pool, config(model="batching", alias_series_ids=False), tmp_path / f"b{i}", model=FunctionModel(batching)) for i in range(3)]
        for result in results:
            records = trace_of(result)
            calls = [c for c in by_kind(records, ToolCallRecord) if c.tool_name == "get_series"]
            assert [c.series_id for c in calls] == ids  # execution order == the order the model issued
            assert [c.call_index for c in calls] == [2, 3, 4, 5, 6]
            assert [s.series_id for s in by_kind(records, SeriesRecord)] == ids
            assert [v.series_id for v in validate_verdicts_file(result.files[VERDICTS_FILE])] == ids
        for other in results[1:]:
            diff = compare_runs(results[0].output_dir, other.output_dir)
            assert diff.identical, diff.differences

    def test_compare_runs_detects_a_difference(self, tiny_pool, tmp_path):
        a = run(tiny_pool, tmp_path, "a")
        b = run(tiny_pool, tmp_path, "b", scripted_options={"max_series": 3})
        assert a.run_id != b.run_id
        diff = compare_runs(a.output_dir, b.output_dir)
        assert not diff.identical and any(VERDICTS_FILE in d for d in diff.differences)

    def test_partial_coverage_is_the_agents_choice(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"max_series": 2})
        end = trace_of(result)[-1]
        assert result.termination.reason == "completed"
        assert end.series.examined == end.series.verdicted == 2 and end.series.in_pool == 5
        assert len(validate_verdicts_file(result.files[VERDICTS_FILE])) == 2

    def test_run_id_reflects_pool_and_config(self, tiny_pool, tmp_path):
        a = run(tiny_pool, tmp_path, "a")
        b = run(tiny_pool, tmp_path, "b", seed=7)
        c = run(tiny_pool.subset(tiny_pool.series_ids[:3]), tmp_path, "c")
        d = run(tiny_pool, tmp_path, "d", order_seed=1)
        assert len({a.run_id, b.run_id, c.run_id, d.run_id}) == 4
        # deployment knobs change where and how fast the model is reached, not what the run is
        deployed = config(provider_base_url="http://x", provider_anonymous=True, request_interval_s=9.0, provider_max_retries=1)
        assert deployed.fingerprint() == config().fingerprint()

    def test_request_pacing_spaces_model_requests(self, tiny_pool, tmp_path):
        import time

        t0 = time.perf_counter()
        result = run(tiny_pool, tmp_path, request_interval_s=0.05)
        elapsed = time.perf_counter() - t0
        end = trace_of(result)[-1]
        assert end.requests == 7 and elapsed >= 0.05 * 6

    def test_run_start_records_the_configuration(self, tiny_pool, tmp_path):
        seen: list[AgentInfo] = []
        inner = ScriptedPolicy()

        def spy(messages, info):
            seen.append(info)
            return inner(messages, info)

        cfg = config(model="spy", temperature=0.0, seed=5, order_seed=3)
        result = run_episode(tiny_pool, cfg, tmp_path / "spy", model=FunctionModel(spy))
        start: RunStartRecord = trace_of(result)[0]
        assert start.model.settings == {"temperature": 0.0, "seed": 5, "parallel_tool_calls": False}
        assert start.prompt.system_prompt_sha256 == prompt_fingerprint(SYSTEM_PROMPT)
        assert start.order_seed == 3 and start.series_id_aliasing is True
        assert start.usage_limits.request_limit == start.usage_limits.tool_calls_limit == 2 * 5 + 8
        assert start.tools == ["get_series", "get_pool_meta_data"]
        assert seen and seen[0].model_settings["seed"] == 5 and seen[0].instructions == SYSTEM_PROMPT.strip()

    def test_seeded_order_is_what_the_agent_walks(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, order_seed=3)
        served = tiny_pool.reordered(3).series_ids
        assert served != tiny_pool.series_ids
        assert [v.series_id for v in validate_verdicts_file(result.files[VERDICTS_FILE])] == served
        again = run(tiny_pool, tmp_path, "again", order_seed=3)
        assert compare_runs(result.output_dir, again.output_dir).identical

    def test_aliasing_hides_pool_ids_from_the_model_only(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, alias_series_ids=True)
        assert result.files[ALIASES_FILE].is_file()
        mapping = json.loads(result.files[ALIASES_FILE].read_text(encoding="utf-8"))
        assert list(mapping) == [f"s{i:04d}" for i in range(1, 6)] and list(mapping.values()) == tiny_pool.series_ids
        raw = result.files[MESSAGES_FILE].read_text(encoding="utf-8")
        assert "SYNTH" not in raw, "a pool id reached the model"
        assert [v.series_id for v in validate_verdicts_file(result.files[VERDICTS_FILE])] == tiny_pool.series_ids
        records = trace_of(result)
        calls = [c for c in by_kind(records, ToolCallRecord) if c.tool_name == "get_series"]
        assert [c.args["series_id"] for c in calls] == list(mapping) and [c.series_id for c in calls] == tiny_pool.series_ids
        assert [e.series_id for e in by_kind(records, VerdictEvent)] == tiny_pool.series_ids
        assert records[0].series_id_aliasing is True and records[-1].series.verdicted == 5

    def test_refuses_to_clobber_an_existing_run(self, tiny_pool, tmp_path):
        run(tiny_pool, tmp_path, "x")
        with pytest.raises(FileExistsError):
            run(tiny_pool, tmp_path, "x")
        run_episode(tiny_pool, config(), tmp_path / "x", overwrite=True)


class TestTermination:
    def test_request_limit_ends_the_episode_gracefully(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, request_limit=6, scripted_options={"misbehave": "never_stop"})
        records = trace_of(result)
        end = records[-1]
        assert result.termination.reason == "usage_limit" and "request_limit" in (end.termination.detail or "")
        assert end.requests == 6 == len(by_kind(records, ModelRequestRecord))
        validate_verdicts_file(result.files[VERDICTS_FILE])

    def test_tool_calls_limit(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, tool_calls_limit=3, scripted_options={"misbehave": "never_stop"})
        end = trace_of(result)[-1]
        assert end.termination.reason == "usage_limit" and "tool_calls_limit" in (end.termination.detail or "")
        assert end.tool_calls.get_series + end.tool_calls.get_pool_meta_data == 3

    def test_budget_projection_stops_before_the_unaffordable_request(self, tiny_pool, tmp_path):
        limit = 12_000
        result = run(tiny_pool, tmp_path, total_tokens_limit=limit, chars_per_token_estimate=1.5, output_allowance_tokens=100)
        records = trace_of(result)
        end = records[-1]
        assert end.termination.reason == "budget" and "not fetched" in (end.termination.detail or "")
        assert end.usage.total_tokens <= limit
        assert 0 < end.series.examined < 5 and end.series.examined_without_verdict == 0
        assert end.series.verdicted == end.series.examined
        # the request that asked for the unaffordable series is recorded, its tool call is not
        last = by_kind(records, ModelRequestRecord)[-1]
        assert last.n_tool_calls == 1 and last.requested_series
        assert all(c.request_index < last.request_index for c in by_kind(records, ToolCallRecord))
        assert end.requests == sum(m["kind"] == "response" for m in messages_of(result))

    def test_budget_stop_collects_the_verdicts_owed_in_a_closing_turn(self, tiny_pool, tmp_path):
        # the stand-in fetches every series before judging any; the cap cuts it off first
        limit = 12_000
        result = run(tiny_pool, tmp_path, total_tokens_limit=limit, chars_per_token_estimate=1.5, output_allowance_tokens=100, scripted_options={"misbehave": "defer_verdicts"})
        records = trace_of(result)
        end = records[-1]
        requests = by_kind(records, ModelRequestRecord)
        assert end.termination.reason == "budget" and "closing turn collected the verdicts" in (end.termination.detail or "")
        assert end.series.examined >= 2 and end.series.examined_without_verdict == 0 and end.series.verdicted == end.series.examined
        assert end.retries.model_dump() == {"tool": 0, "output": 0, "closing": 1}
        assert requests[-1].retry_kind == "closing" and requests[-1].verdicts_accepted == end.series.verdicted
        assert end.usage.total_tokens <= limit
        skipped = [c for c in by_kind(records, ToolCallRecord) if not c.ok]
        assert len(skipped) == 1 and "not executed" in (skipped[0].error or "") and skipped[0].request_index == requests[-2].request_index
        messages = messages_of(result)
        closing_request = messages[-2]
        kinds = [p["part_kind"] for p in closing_request["parts"]]
        assert kinds == ["tool-return", "user-prompt"] and "budget is exhausted" in closing_request["parts"][0]["content"]
        assert end.requests == sum(m["kind"] == "response" for m in messages)
        validate_verdicts_file(result.files[VERDICTS_FILE])

    def test_closing_turn_is_skipped_when_it_would_not_fit(self, tiny_pool, tmp_path):
        # a large reply allowance makes the stop fire while the context is still small, and
        # leaves no room for the closing request either
        # (with the stand-in's fixed token estimates: the stop fires on the third response,
        # after one series, and 6,300 + the context no longer fit under the cap)
        result = run(tiny_pool, tmp_path, total_tokens_limit=12_000, chars_per_token_estimate=1.5, output_allowance_tokens=6_300, scripted_options={"misbehave": "defer_verdicts"})
        end = trace_of(result)[-1]
        assert end.termination.reason == "budget" and "no closing turn" in (end.termination.detail or ""), end.termination.detail
        assert end.retries.closing == 0 and end.series.examined_without_verdict == end.series.examined >= 1
        assert end.usage.total_tokens <= 12_000

    def test_closing_turn_never_executes_a_tool(self, tiny_pool, tmp_path):
        inner = ScriptedPolicy(misbehave="defer_verdicts")

        def greedy(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) and "budget is exhausted" in str(p.content) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": tiny_pool.series_ids[-1]})], usage=usage())
            return inner(messages, info)

        cfg = config(model="greedy", total_tokens_limit=12_000, chars_per_token_estimate=1.5, output_allowance_tokens=100)
        result = run_episode(tiny_pool, cfg, tmp_path / "g", model=FunctionModel(greedy))
        records = trace_of(result)
        end = records[-1]
        assert end.termination.reason == "budget" and "asked for another tool call" in (end.termination.detail or "")
        executed = [c for c in by_kind(records, ToolCallRecord) if c.ok]
        assert all(c.request_index < by_kind(records, ModelRequestRecord)[-1].request_index for c in executed)
        assert end.series.examined_without_verdict == end.series.examined >= 1 and end.retries.closing == 1

    def test_context_window_projection(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, context_tokens_limit=6_000, chars_per_token_estimate=1.5, output_allowance_tokens=100)
        end = trace_of(result)[-1]
        assert end.termination.reason == "budget" and "context_tokens_limit" in (end.termination.detail or "")
        assert end.series.examined_without_verdict == 0 and end.series.examined >= 1

    def test_batch_beyond_the_window_is_served_in_issue_order(self, tiny_pool, tmp_path):
        ids = tiny_pool.series_ids
        sizes = [len(get_series(tiny_pool, s)) for s in ids]

        def batching(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_pool_meta_data", {})], usage=usage())
            if any(isinstance(p, M.ToolReturnPart) and p.tool_name == "get_pool_meta_data" for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": s}) for s in ids], usage=usage())
            served = [json.loads(p.content)["series_id"] for p in last if isinstance(p, M.ToolReturnPart) and not p.content.startswith("Error")]
            lines = [json.dumps({"series_id": s, "verdict": "normal"}) for s in served]
            return M.ModelResponse(parts=[M.TextPart("\n".join(lines + [f"DONE {len(served)}"]))], usage=usage())

        # usage() reports one input and one output token; at one character per token the request
        # after the batch needs 2 + payload tokens, so this window admits exactly two series.
        cfg = config(model="batching", alias_series_ids=False, context_tokens_limit=2 + sizes[0] + sizes[1] + 10, chars_per_token_estimate=1.0, output_allowance_tokens=10)
        result = run_episode(tiny_pool, cfg, tmp_path / "batch", model=FunctionModel(batching))
        records = trace_of(result)
        end = records[-1]
        calls = [c for c in by_kind(records, ToolCallRecord) if c.tool_name == "get_series"]
        assert [c.ok for c in calls] == [True, True, False, False, False]
        assert all("not retrieved" in (c.error or "") and c.series_id in (c.error or "") for c in calls[2:])
        assert end.termination.reason == "completed" and end.tool_calls.failed == 3 and end.tool_calls.get_series == 5
        assert end.series.examined == 2 and [v.series_id for v in validate_verdicts_file(result.files[VERDICTS_FILE])] == ids[:2]
        returns = [p for m in messages_of(result) for p in m["parts"] if p.get("part_kind") == "tool-return" and p.get("tool_name") == "get_series"]
        assert [p["content"].startswith("Error: not retrieved") for p in returns] == [False, False, True, True, True]

    def test_token_limit_after_a_response_keeps_that_response(self, tiny_pool, tmp_path):
        sid = tiny_pool.series_ids[0]

        def policy(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_pool_meta_data", {})], usage=usage())
            if any(isinstance(p, M.ToolReturnPart) and p.tool_name == "get_pool_meta_data" for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": sid})], usage=usage())
            return M.ModelResponse(
                parts=[M.TextPart(json.dumps({"series_id": sid, "verdict": "anomalous", "explanation": "index 120 = 25"}) + "\nDONE 1")],
                usage=RequestUsage(input_tokens=6_500, output_tokens=40, details={"reasoning_tokens": 7}),
                model_name="verbose",
                finish_reason="stop",
            )

        cfg = config(model="verbose", alias_series_ids=False, total_tokens_limit=6_000, chars_per_token_estimate=100.0, output_allowance_tokens=10)
        result = run_episode(tiny_pool, cfg, tmp_path / "v", model=FunctionModel(policy))
        records = trace_of(result)
        end = records[-1]
        requests = by_kind(records, ModelRequestRecord)
        assert end.termination.reason == "usage_limit" and "total_tokens_limit" in (end.termination.detail or "")
        assert end.requests == len(requests) == 3
        assert requests[-1].usage.input_tokens == 6_500 and requests[-1].usage.details == {"reasoning_tokens": 7}
        assert requests[-1].model_name is not None and requests[-1].verdicts_accepted == 1
        assert end.usage.details == {"reasoning_tokens": 7} and end.usage.total_tokens > 6_000
        assert end.series.verdicted == 1 and end.series.examined_without_verdict == 0
        assert [v.verdict for v in validate_verdicts_file(result.files[VERDICTS_FILE])] == ["anomalous"]
        assert sum(m["kind"] == "response" for m in messages_of(result)) == 3

    def test_transient_http_errors_are_retried_then_given_up(self, tiny_pool, tmp_path):
        inner = ScriptedPolicy()
        calls = {"n": 0}

        def rate_limited(messages, info):
            calls["n"] += 1
            if calls["n"] in (2, 3):  # two 429s in a row on the second request
                raise ModelHTTPError(status_code=429, model_name="tier", body={"message": "API rate limit exceeded"})
            if calls["n"] == 5:
                raise ModelHTTPError(status_code=503, model_name="tier", body="busy")
            return inner(messages, info)

        cfg = config(model="tier", transient_retries=3, rate_limit_wait_s=0.01)
        result = run_episode(tiny_pool, cfg, tmp_path / "t", model=FunctionModel(rate_limited))
        end = trace_of(result)[-1]
        assert end.termination.reason == "completed" and end.requests == 7 and calls["n"] == 10
        assert end.series.verdicted == 5

        calls["n"] = 0
        cfg = config(model="tier", transient_retries=1, rate_limit_wait_s=0.01)
        result = run_episode(tiny_pool, cfg, tmp_path / "u", model=FunctionModel(rate_limited))
        end = trace_of(result)[-1]
        assert end.termination.reason == "model_error" and "429" in (end.termination.detail or "")

    def test_model_error_is_recorded_and_outputs_survive(self, tiny_pool, tmp_path):
        inner = ScriptedPolicy()
        calls = {"n": 0}

        def flaky(messages, info):
            calls["n"] += 1
            if calls["n"] == 4:
                raise ModelHTTPError(status_code=400, model_name="flaky", body={"error": "context length exceeded"})
            return inner(messages, info)

        result = run_episode(tiny_pool, config(model="flaky"), tmp_path / "flaky", model=FunctionModel(flaky))
        records = trace_of(result)
        end = records[-1]
        assert end.termination.reason == "model_error" and "context length exceeded" in (end.termination.detail or "")
        assert end.requests == 3
        assert [v.series_id for v in validate_verdicts_file(result.files[VERDICTS_FILE])] == tiny_pool.series_ids[:1]

    def test_cancellation_is_an_interrupted_run_with_complete_outputs(self, tiny_pool, tmp_path):
        inner = ScriptedPolicy()
        calls = {"n": 0}

        def cancelled(messages, info):
            calls["n"] += 1
            if calls["n"] == 4:
                raise asyncio.CancelledError()
            return inner(messages, info)

        result = run_episode(tiny_pool, config(model="cancelled"), tmp_path / "c", model=FunctionModel(cancelled))
        records = trace_of(result)
        end = records[-1]
        assert end.termination.reason == "interrupted" and end.termination.detail == "CancelledError"
        for name in (VERDICTS_FILE, TRACE_FILE, MESSAGES_FILE, SUMMARY_FILE):
            assert result.files[name].is_file()
        assert end.requests == 3 and len(validate_verdicts_file(result.files[VERDICTS_FILE])) == 1

    def test_exhausted_output_retries_is_a_protocol_error(self, tiny_pool, tmp_path):
        sid = tiny_pool.series_ids[0]

        def stubborn(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": sid})], usage=usage())
            return M.ModelResponse(parts=[M.TextPart("DONE 1")], usage=usage())

        result = run_episode(tiny_pool, config(model="stubborn", alias_series_ids=False, output_retries=1), tmp_path / "s", model=FunctionModel(stubborn))
        records = trace_of(result)
        end = records[-1]
        assert end.termination.reason == "protocol_error" and "retries" in (end.termination.detail or "")
        assert end.retries.model_dump() == {"tool": 0, "output": 1, "closing": 0}
        assert end.series.examined_without_verdict == 1 and end.series.verdicted == 0
        assert [r.retry_kind for r in by_kind(records, ModelRequestRecord)] == [None, None, "output"]


class TestProtocolCounterMeasures:
    def test_missing_verdict_triggers_one_retry(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"misbehave": "skip_last_verdict"})
        records = trace_of(result)
        end = records[-1]
        assert end.termination.reason == "completed" and end.retries.model_dump() == {"tool": 0, "output": 1, "closing": 0}
        assert end.series.verdicted == 5 and end.series.examined_without_verdict == 0
        retry_requests = [r for r in by_kind(records, ModelRequestRecord) if r.is_retry]
        assert len(retry_requests) == 1 and retry_requests[0].retry_kind == "output" and retry_requests[0].verdicts_accepted == 1
        retry_parts = [p for m in messages_of(result) for p in m["parts"] if p["part_kind"] == "retry-prompt"]
        assert len(retry_parts) == 1 and "still have no verdict" in retry_parts[0]["content"]

    def test_retry_message_speaks_the_ids_the_model_knows(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, alias_series_ids=True, scripted_options={"misbehave": "skip_last_verdict"})
        retry_parts = [p for m in messages_of(result) for p in m["parts"] if p["part_kind"] == "retry-prompt"]
        assert len(retry_parts) == 1 and "s0005" in retry_parts[0]["content"] and "SYNTH" not in retry_parts[0]["content"]
        assert trace_of(result)[-1].series.verdicted == 5

    def test_enforcement_can_be_switched_off(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, enforce_verdicts=False, scripted_options={"misbehave": "skip_last_verdict"})
        end = trace_of(result)[-1]
        assert end.termination.reason == "completed" and end.retries.output == 0
        assert end.series.examined == 5 and end.series.verdicted == 4 and end.series.examined_without_verdict == 1

    def test_blind_verdict_is_discarded(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"max_series": 3, "misbehave": "blind_verdict"})
        records = trace_of(result)
        end = records[-1]
        rejected = [e for e in by_kind(records, VerdictEvent) if not e.accepted]
        assert len(rejected) == 1 and rejected[0].reason == "not_examined" and not rejected[0].examined
        assert end.verdicts.rejected == 1 and end.series.verdicted == 3
        assert rejected[0].series_id not in {v.series_id for v in validate_verdicts_file(result.files[VERDICTS_FILE])}

    def test_unknown_id_verdict_is_discarded(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"max_series": 2, "misbehave": "unknown_id"})
        records = trace_of(result)
        rejected = [e for e in by_kind(records, VerdictEvent) if not e.accepted]
        assert [e.reason for e in rejected] == ["unknown_series_id"]
        assert records[-1].series.verdicted == 2

    def test_malformed_verdict_line_is_reported_then_recovered(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"misbehave": "bad_json"})
        records = trace_of(result)
        end = records[-1]
        rejected = [e for e in by_kind(records, VerdictEvent) if not e.accepted]
        assert len(rejected) == 1 and rejected[0].reason.startswith("invalid_record")
        assert end.termination.reason == "completed" and end.retries.output == 1 and end.series.verdicted == 5

    def test_quitting_immediately_is_allowed(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"misbehave": "quit_early"})
        end = trace_of(result)[-1]
        assert end.termination.reason == "completed" and end.retries.output == 0
        assert end.series.examined == 0 and end.series.verdicted == 0
        assert validate_verdicts_file(result.files[VERDICTS_FILE]) == []

    def test_malformed_series_reaches_the_model_as_text(self, malformed_pool, tmp_path):
        result = run(malformed_pool, tmp_path)
        records = trace_of(result)
        end = records[-1]
        failed = [c for c in by_kind(records, ToolCallRecord) if not c.ok]
        assert [c.series_id for c in failed] == ["M__nan_full", "M__gap_full"]
        assert all("malformed" in (c.error or "") for c in failed)
        series = {s.series_id: s for s in by_kind(records, SeriesRecord)}
        assert series["M__nan_full"].failed_get_series_calls == 1 and not series["M__nan_full"].examined
        assert end.tool_calls.failed == 2 and end.series.examined == 1 and end.series.verdicted == 1
        assert end.termination.reason == "completed"

    def test_invalid_tool_arguments_and_unknown_tools_are_traced(self, tiny_pool, tmp_path):
        sid = tiny_pool.series_ids[0]
        step = {"n": 0}

        def sloppy(messages, info):
            step["n"] += 1
            if step["n"] == 1:
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": 123})], usage=usage())
            if step["n"] == 2:
                return M.ModelResponse(parts=[M.ToolCallPart("cheap_scan", {})], usage=usage())
            if step["n"] == 3:
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": sid})], usage=usage())
            return M.ModelResponse(parts=[M.TextPart(json.dumps({"series_id": sid, "verdict": "normal"}) + "\nDONE 1")], usage=usage())

        result = run_episode(tiny_pool, config(model="sloppy", alias_series_ids=False, tool_retries=2), tmp_path / "sloppy", model=FunctionModel(sloppy))
        records = trace_of(result)
        end = records[-1]
        calls = by_kind(records, ToolCallRecord)
        assert end.termination.reason == "completed" and end.series.verdicted == 1
        assert [(c.tool_name, c.ok) for c in calls] == [("get_series", False), ("cheap_scan", False), ("get_series", True)]
        assert end.tool_calls.failed == 2 and end.tool_calls.get_series == 2
        assert end.retries.model_dump() == {"tool": 2, "output": 0, "closing": 0}
        assert [r.retry_kind for r in by_kind(records, ModelRequestRecord)] == [None, "tool", "tool", None]
