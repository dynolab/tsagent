"""Tool registration and behaviour of get_series / get_pool_meta_data (spec 02, FR1-FR3).

No LLM is involved: the agent is driven by pydantic-ai's FunctionModel / TestModel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic_ai import messages as M
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from src.baseline_agent import TOOL_NAMES, Pool, build_agent, get_pool_meta_data, get_series, read_split_ids, registered_tool_names
from src.baseline_agent.agent import EpisodeDeps
from src.baseline_agent.ids import IdAliases, PlainIds
from src.baseline_agent.models import build_model
from src.baseline_agent.prompt import SYSTEM_PROMPT, USER_PROMPT
from src.baseline_agent.runner import EpisodeConfig
from src.baseline_agent.tools import is_error
from src.baseline_agent.trace import TraceRecorder
from src.baseline_agent.verdicts import VerdictCollector
from src.baseline_agent.writers import JsonlWriter

FORBIDDEN_IN_TOOL_DESCRIPTIONS = ("anomal", "detect", "score", "statistic", "scan", "outlier")


def make_deps(pool: Pool, tmp_path: Path, **kwargs) -> tuple[EpisodeDeps, JsonlWriter]:
    writer = JsonlWriter(tmp_path / "trace.jsonl")
    deps = EpisodeDeps(pool=pool, recorder=TraceRecorder(writer, pool), collector=VerdictCollector(), **kwargs)
    return deps, writer


def capture_agent_info(tiny_pool, tmp_path, **agent_kwargs) -> tuple[AgentInfo, list[M.ModelMessage]]:
    seen: list[AgentInfo] = []
    history: list[M.ModelMessage] = []

    def policy(messages, info: AgentInfo) -> M.ModelResponse:
        seen.append(info)
        history[:] = messages
        return M.ModelResponse(parts=[M.TextPart("DONE 0")], usage=RequestUsage(input_tokens=1, output_tokens=1))

    agent = build_agent(FunctionModel(policy), **agent_kwargs)
    deps, writer = make_deps(tiny_pool, tmp_path)
    agent.run_sync(USER_PROMPT, deps=deps)
    writer.close()
    assert len(seen) == 1
    return seen[0], history


# --- FR3: exactly two tools, greedy decoding, the prompt --------------------------------


class TestRegistration:
    def test_exactly_the_two_spec_tools_are_registered(self):
        agent = build_agent(build_model("test"))
        assert sorted(registered_tool_names(agent)) == sorted(TOOL_NAMES)
        assert set(TOOL_NAMES) == {"get_series", "get_pool_meta_data"}

    def test_model_sees_two_function_tools_and_no_output_tool(self, tiny_pool, tmp_path):
        info, _ = capture_agent_info(tiny_pool, tmp_path)
        assert sorted(t.name for t in info.function_tools) == sorted(TOOL_NAMES)
        assert info.output_tools == []
        assert info.allow_text_output is True

    def test_tool_schemas_promise_retrieval_only(self, tiny_pool, tmp_path):
        info, _ = capture_agent_info(tiny_pool, tmp_path)
        by_name = {t.name: t for t in info.function_tools}
        assert list(by_name["get_series"].parameters_json_schema["properties"]) == ["series_id"]
        assert by_name["get_series"].parameters_json_schema["properties"]["series_id"]["type"] == "string"
        assert by_name["get_pool_meta_data"].parameters_json_schema.get("properties", {}) == {}
        for tool in by_name.values():
            description = (tool.description or "").lower()
            assert description, f"{tool.name} has no description"
            for word in FORBIDDEN_IN_TOOL_DESCRIPTIONS:
                assert word not in description, f"{tool.name} description mentions {word!r}"
            assert tool.sequential is True

    def test_decoding_settings_and_prompt_reach_the_model(self, tiny_pool, tmp_path):
        config = EpisodeConfig(model="unused", temperature=0.0, seed=42, parallel_tool_calls=False)
        info, history = capture_agent_info(tiny_pool, tmp_path, model_settings=config.model_settings())
        assert info.model_settings is not None
        assert info.model_settings["temperature"] == 0.0
        assert info.model_settings["seed"] == 42
        assert info.model_settings["parallel_tool_calls"] is False
        assert info.instructions == SYSTEM_PROMPT.strip()  # pydantic-ai trims the trailing newline
        requests = [m for m in history if isinstance(m, M.ModelRequest)]
        assert requests and requests[-1].instructions == SYSTEM_PROMPT.strip()
        assert "label" not in SYSTEM_PROMPT.lower() and "%" not in SYSTEM_PROMPT


# --- FR1: get_series ------------------------------------------------------------------


class TestGetSeries:
    def test_returns_the_full_raw_series(self, tiny_pool):
        sid = tiny_pool.series_ids[2]
        series = tiny_pool.get(sid)
        text = get_series(tiny_pool, sid)
        assert not is_error(text)
        doc = json.loads(text)
        assert set(doc) == {"series_id", "length", "time_index", "value"}
        assert doc["series_id"] == sid
        assert doc["length"] == len(series) == len(doc["value"]) == len(doc["time_index"])
        assert doc["time_index"] == list(range(len(series)))
        assert doc["value"] == series.value.tolist()
        assert doc["value"][120] == 25.0  # the spike is delivered raw, not smoothed away

    def test_no_label_or_metadata_leaks(self, tiny_parquet):
        pool = Pool.from_parquet(tiny_parquet)
        assert pool.name == "T1"
        for sid in pool.series_ids:
            text = get_series(pool, sid)
            assert "label" not in text
            assert set(json.loads(text)) == {"series_id", "length", "time_index", "value"}
        assert "label" not in get_pool_meta_data(pool)
        assert not hasattr(pool.get(pool.series_ids[0]), "label")

    def test_unknown_id_gives_a_textual_error_not_an_exception(self, tiny_pool):
        text = get_series(tiny_pool, "T1__SYNTH__does-not-exist_full")
        assert is_error(text)
        assert "unknown series_id" in text and "get_pool_meta_data" in text
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_blank_id_is_an_error(self, tiny_pool, bad):
        assert is_error(get_series(tiny_pool, bad))

    def test_malformed_series_give_textual_errors(self, malformed_pool):
        assert not is_error(get_series(malformed_pool, "M__a_full"))
        nan = get_series(malformed_pool, "M__nan_full")
        assert is_error(nan) and "non-finite" in nan and "malformed" in nan
        gap = get_series(malformed_pool, "M__gap_full")
        assert is_error(gap) and "not contiguous" in gap

    def test_wire_format_is_compact_and_deterministic(self, tiny_pool):
        sid = tiny_pool.series_ids[0]
        a, b = get_series(tiny_pool, sid), get_series(tiny_pool, sid)
        assert a == b
        assert ": " not in a and ", " not in a
        assert a.startswith('{"series_id":"')

    def test_value_precision_is_opt_in(self, tiny_pool):
        sid = tiny_pool.series_ids[0]
        raw = json.loads(get_series(tiny_pool, sid))["value"]
        rounded = json.loads(get_series(tiny_pool, sid, value_precision=2))["value"]
        assert rounded == [round(v, 2) for v in raw]
        assert rounded != raw

    def test_integer_pools_stay_integers_and_narrow_floats_are_refused(self):
        ints = pd.DataFrame({"series_id": ["i"] * 4, "time_index": range(4), "value": np.array([3, 4, 3, 9], dtype=np.int32)})
        pool = Pool.from_frame(ints, "I")
        doc = json.loads(get_series(pool, "i"))
        assert doc["value"] == [3, 4, 3, 9] and all(isinstance(v, int) for v in doc["value"])
        assert json.loads(get_series(pool, "i", value_precision=1))["value"] == [3, 4, 3, 9]
        f32 = pd.DataFrame({"series_id": ["f"] * 3, "time_index": range(3), "value": np.array([0.1, 0.2, 0.3], dtype=np.float32)})
        with pytest.raises(ValueError, match="float64 or integer"):
            Pool.from_frame(f32, "F")

    def test_label_argument_names_the_series_for_the_model(self, tiny_pool):
        sid = tiny_pool.series_ids[0]
        doc = json.loads(get_series(tiny_pool, sid, label="s0001"))
        assert doc["series_id"] == "s0001" and sid not in json.dumps(doc)


# --- FR2: get_pool_meta_data ------------------------------------------------------------


class TestGetPoolMetaData:
    def test_lists_every_id_in_pool_order(self, tiny_pool):
        doc = json.loads(get_pool_meta_data(tiny_pool))
        assert set(doc) == {"pool", "n_series", "series_ids"}
        assert doc["pool"] == "T1"
        assert doc["n_series"] == len(tiny_pool) == 5
        assert doc["series_ids"] == tiny_pool.series_ids
        assert len(set(doc["series_ids"])) == len(doc["series_ids"])

    def test_served_ids_can_be_aliases(self, tiny_pool):
        aliases = IdAliases(tiny_pool.series_ids)
        doc = json.loads(get_pool_meta_data(tiny_pool, series_ids=aliases.served))
        assert doc["series_ids"] == ["s0001", "s0002", "s0003", "s0004", "s0005"]
        assert doc["n_series"] == 5

    def test_split_selection_reads_only_ids_and_split(self, tiny_pool, tmp_path):
        meta = pd.DataFrame(
            {
                "series_id": tiny_pool.series_ids,
                "y_i": [0, 0, 1, 0, 1],
                "split": ["dev", "test", "dev", "test", "test"],
            }
        )
        path = tmp_path / "T1_metadata.parquet"
        meta.to_parquet(path, index=False)
        dev = read_split_ids(path, "dev")
        assert dev == [tiny_pool.series_ids[0], tiny_pool.series_ids[2]]
        sub = tiny_pool.subset(dev)
        assert sub.series_ids == dev
        with pytest.raises(ValueError):
            read_split_ids(path, "validation")


# --- the agent-facing wrappers ----------------------------------------------------------


class TestAgentWrappers:
    def test_calls_are_timed_traced_and_mark_examination(self, tiny_pool, tmp_path):
        sid = tiny_pool.series_ids[1]

        def policy(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_pool_meta_data", {})], usage=RequestUsage(input_tokens=1, output_tokens=1))
            if any(isinstance(p, M.ToolReturnPart) and p.tool_name == "get_pool_meta_data" for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": sid}), M.ToolCallPart("get_series", {"series_id": "nope"})], usage=RequestUsage(input_tokens=1, output_tokens=1))
            return M.ModelResponse(parts=[M.TextPart(json.dumps({"series_id": sid, "verdict": "normal"}) + "\nDONE 1")], usage=RequestUsage(input_tokens=1, output_tokens=1))

        agent = build_agent(FunctionModel(policy))
        deps, writer = make_deps(tiny_pool, tmp_path)
        result = agent.run_sync(USER_PROMPT, deps=deps)
        writer.close()
        assert "DONE 1" in result.output
        assert dict(deps.collector.examined) == {sid: 1}
        calls = deps.recorder.tool_calls
        assert [c.tool_name for c in calls] == ["get_pool_meta_data", "get_series", "get_series"]
        ok, bad = calls[1], calls[2]
        assert ok.ok and ok.series_id == sid and ok.n_points == len(tiny_pool.get(sid)) and ok.error is None
        assert ok.payload_chars == len(get_series(tiny_pool, sid))
        assert ok.duration_ms >= 0 and ok.request_index == 2 and ok.tool_call_id
        assert not bad.ok and bad.series_id == "nope" and "unknown series_id" in (bad.error or "")
        lines = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [l["kind"] for l in lines] == ["tool_call"] * 3

    def test_stray_tool_arguments_are_ignored_not_refused(self, tiny_pool, tmp_path):
        sid = tiny_pool.series_ids[0]

        def policy(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_pool_meta_data", {"": ""})], usage=RequestUsage(input_tokens=1, output_tokens=1))
            if any(isinstance(p, M.ToolReturnPart) and p.tool_name == "get_pool_meta_data" for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": sid, "format": "json"})], usage=RequestUsage(input_tokens=1, output_tokens=1))
            return M.ModelResponse(parts=[M.TextPart(json.dumps({"series_id": sid, "verdict": "normal"}) + "\nDONE 1")], usage=RequestUsage(input_tokens=1, output_tokens=1))

        agent = build_agent(FunctionModel(policy))
        deps, writer = make_deps(tiny_pool, tmp_path)
        agent.run_sync(USER_PROMPT, deps=deps)
        writer.close()
        calls = deps.recorder.tool_calls
        assert [(c.tool_name, c.ok, c.args) for c in calls] == [
            ("get_pool_meta_data", True, {"": ""}),
            ("get_series", True, {"series_id": sid, "format": "json"}),
        ]
        assert deps.recorder.retries == {"tool": 0, "output": 0, "closing": 0} and dict(deps.collector.examined) == {sid: 1}

    def test_aliased_wrappers_translate_both_ways(self, tiny_pool, tmp_path):
        aliases = IdAliases(tiny_pool.series_ids)
        real = tiny_pool.series_ids[3]
        alias = aliases.display(real)

        def policy(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_pool_meta_data", {})], usage=RequestUsage(input_tokens=1, output_tokens=1))
            if any(isinstance(p, M.ToolReturnPart) and p.tool_name == "get_pool_meta_data" for p in last):
                ids = json.loads(last[0].content)["series_ids"]
                assert ids == aliases.served and not any("SYNTH" in i for i in ids)
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": alias}), M.ToolCallPart("get_series", {"series_id": real})], usage=RequestUsage(input_tokens=1, output_tokens=1))
            returns = [p for p in last if isinstance(p, M.ToolReturnPart)]
            assert json.loads(returns[0].content)["series_id"] == alias
            assert is_error(returns[1].content)  # the real id is not a name the model was given
            return M.ModelResponse(parts=[M.TextPart(json.dumps({"series_id": alias, "verdict": "normal"}) + "\nDONE 1")], usage=RequestUsage(input_tokens=1, output_tokens=1))

        agent = build_agent(FunctionModel(policy))
        deps, writer = make_deps(tiny_pool, tmp_path, ids=aliases)
        agent.run_sync(USER_PROMPT, deps=deps)
        writer.close()
        assert dict(deps.collector.examined) == {real: 1}
        assert [r.series_id for r in deps.collector.records()] == [real]
        calls = deps.recorder.tool_calls
        assert calls[1].args == {"series_id": alias} and calls[1].series_id == real and calls[1].ok
        assert calls[2].args == {"series_id": real} and not calls[2].ok

    def test_pool_from_parquet_rejects_missing_columns(self, tmp_path):
        frame = pd.DataFrame({"series_id": ["a"] * 3, "value": np.arange(3.0)})
        path = tmp_path / "broken.parquet"
        frame.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="time_index"):
            Pool.from_parquet(path)

    def test_pool_fingerprint_tracks_content(self):
        frame = pd.DataFrame(
            {"series_id": ["x"] * 3 + ["y"] * 2, "time_index": [0, 1, 2, 0, 1], "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )
        a = Pool.from_frame(frame, "P")
        b = Pool.from_frame(frame.copy(), "P")
        frame.loc[0, "value"] = 1.5
        c = Pool.from_frame(frame, "P")
        assert a.fingerprint == b.fingerprint != c.fingerprint
        assert a.series_ids == ["x", "y"] and a.subset(["y"]).series_ids == ["y"]
        with pytest.raises(ValueError, match="duplicate"):
            Pool("D", [a.get("x"), a.get("x")])

    def test_pool_rejects_padded_ids_and_reorders_deterministically(self, tiny_pool):
        padded = pd.DataFrame({"series_id": [" p "] * 3, "time_index": [0, 1, 2], "value": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="whitespace"):
            Pool.from_frame(padded, "P")
        assert tiny_pool.reordered(None) is tiny_pool
        once, twice = tiny_pool.reordered(7), tiny_pool.reordered(7)
        assert once.series_ids == twice.series_ids
        assert sorted(once.series_ids) == sorted(tiny_pool.series_ids)
        assert once.series_ids != tiny_pool.series_ids
        assert once.series_ids != tiny_pool.reordered(8).series_ids
        assert once.get(once.series_ids[0]) is tiny_pool.get(once.series_ids[0])


class TestModelResolution:
    def test_openai_compatible_endpoint_with_and_without_credentials(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        from src.baseline_agent.models import build_model

        model = build_model("openai-chat:gpt-oss-20b", base_url="http://localhost:1/v1", api_key="k", timeout_s=7.0, max_retries=2)
        assert isinstance(model, OpenAIChatModel) and model.model_name == "gpt-oss-20b"
        assert str(model.client.base_url).startswith("http://localhost:1/v1") and model.client.max_retries == 2
        anonymous = build_model("gpt-oss-20b", base_url="http://localhost:1/v1", anonymous=True)
        assert isinstance(anonymous, OpenAIChatModel) and anonymous.model_name == "gpt-oss-20b"

    def test_anonymous_client_strips_the_authorization_header(self):
        import asyncio

        import httpx

        from src.baseline_agent.models import anonymous_http_client

        seen: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(dict(request.headers))
            return httpx.Response(200, json={"ok": True})

        async def go():
            async with anonymous_http_client(5.0, transport=httpx.MockTransport(handler)) as client:
                await client.get("http://x/v1/models", headers={"Authorization": "Bearer unused", "X-Other": "kept"})

        asyncio.run(go())
        assert seen and "authorization" not in seen[0] and seen[0]["x-other"] == "kept"


class TestIdViews:
    def test_plain_ids_resolve_exact_then_trimmed(self, tiny_pool):
        ids = PlainIds(tiny_pool.series_ids)
        sid = tiny_pool.series_ids[0]
        assert ids.served == tiny_pool.series_ids and ids.display(sid) == sid
        assert ids.resolve(sid) == sid and ids.resolve(f" {sid}\n") == sid and ids.resolve("nope") is None
        assert ids.mapping() is None and ids.aliased is False

    def test_aliases_are_opaque_and_round_trip(self, tiny_pool):
        ids = IdAliases(tiny_pool.series_ids)
        assert ids.served == [f"s{i:04d}" for i in range(1, 6)] and ids.aliased is True
        for sid in tiny_pool.series_ids:
            alias = ids.display(sid)
            assert ids.resolve(alias) == sid and ids.resolve(f"  {alias} ") == sid
            assert "SYNTH" not in alias
        assert ids.resolve(tiny_pool.series_ids[0]) is None and ids.resolve("s9999") is None
        assert list(ids.mapping().items())[0] == ("s0001", tiny_pool.series_ids[0])
        wide = IdAliases([f"id{i}" for i in range(12000)])
        assert wide.served[0] == "s00001" and wide.served[-1] == "s12000"
