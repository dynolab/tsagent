"""A scripted stand-in for the LLM: dry runs and harness tests without an API.

``ScriptedPolicy`` is a pydantic-ai ``FunctionModel`` callback. It plays the agent's part
deterministically - discover the pool, retrieve series one by one in pool order, judge
each with a fixed robust-z rule, stop after ``max_series`` - so the whole pipeline (tool
registration, verdict protocol, trace, writers, determinism check) can be exercised end to
end on a real pool. Its verdicts are meaningless for RQ1: it is the lower layer being
mocked, not a baseline result, and every run it produces says ``model_name = "scripted"``.

The ``misbehave`` switch makes it break the protocol on purpose so the harness's
counter-measures (discarded verdicts, the missing-verdict retry, the request cap) can be
tested.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai import messages as M
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from .tools import is_error
from .verdicts import parse_verdicts

Misbehaviour = Literal["skip_last_verdict", "blind_verdict", "unknown_id", "bad_json", "never_stop", "quit_early", "defer_verdicts"]

MODEL_NAME = "scripted"


@dataclass
class _State:
    ids: list[str] | None
    fetched: dict[str, list[float]]
    failed: list[str]
    verdicted: set[str]
    last_returns: list[M.ToolReturnPart]
    retry: M.RetryPromptPart | None
    closing: bool  # the harness asked for the verdicts owed after a budget stop
    n_responses: int


class ScriptedPolicy:
    def __init__(
        self,
        *,
        max_series: int | None = None,
        z_threshold: float = 8.0,
        misbehave: Misbehaviour | None = None,
        chars_per_token: int = 4,
    ):
        self.max_series = max_series
        self.z_threshold = z_threshold
        self.misbehave = misbehave
        self.chars_per_token = chars_per_token

    # --- pydantic-ai entry point ------------------------------------------------------

    def __call__(self, messages: list[M.ModelMessage], info: AgentInfo) -> M.ModelResponse:
        state = self._read(messages)
        parts = self._decide(state)
        usage = RequestUsage(
            input_tokens=self._estimate(_history_text(messages)),
            output_tokens=self._estimate(_parts_text(parts)),
        )
        return M.ModelResponse(parts=parts, usage=usage, model_name=MODEL_NAME, finish_reason="stop")

    # --- policy -----------------------------------------------------------------------

    def _decide(self, s: _State) -> list[M.ModelResponsePart]:
        if s.ids is None:
            return [M.ToolCallPart("get_pool_meta_data", {})]
        if s.retry is not None or s.closing:
            pending = [sid for sid in s.fetched if sid not in s.verdicted]
            lines = [self._verdict_line(sid, s.fetched[sid]) for sid in pending]
            return [M.TextPart("\n".join(lines + [f"DONE {len(s.fetched)}"]))]

        lines: list[str] = []
        if self.misbehave != "defer_verdicts":
            for ret in s.last_returns:
                if ret.tool_name != "get_series":
                    continue
                sid = _series_id_of(ret)
                if sid is None or sid not in s.fetched or sid in s.verdicted:
                    continue
                lines.append(self._verdict_line(sid, s.fetched[sid]))

        if self.misbehave == "quit_early" and s.n_responses >= 1:
            return [M.TextPart("DONE 0")]

        budget = self.max_series if self.max_series is not None else len(s.ids)
        attempted = set(s.fetched) | set(s.failed)
        remaining = [sid for sid in s.ids if sid not in attempted]
        done = len(attempted) >= budget or not remaining

        if self.misbehave == "never_stop":
            return [M.TextPart("\n".join(lines)) if lines else M.TextPart("..."), M.ToolCallPart("get_series", {"series_id": s.ids[0]})]

        if not done:
            next_id = remaining[0]
            parts: list[M.ModelResponsePart] = []
            if lines:
                parts.append(M.TextPart("\n".join(lines)))
            parts.append(M.ToolCallPart("get_series", {"series_id": next_id}))
            return parts

        if self.misbehave == "defer_verdicts":
            lines = [self._verdict_line(sid, s.fetched[sid]) for sid in s.fetched if sid not in s.verdicted]
        if self.misbehave == "skip_last_verdict" and lines:
            lines = lines[:-1]
        if self.misbehave == "blind_verdict" and remaining:
            lines.append(json.dumps({"series_id": remaining[0], "verdict": "normal"}))
        if self.misbehave == "unknown_id":
            lines.append(json.dumps({"series_id": "NOT__A__SERIES_full", "verdict": "normal"}))
        if self.misbehave == "bad_json" and lines:
            lines[-1] = lines[-1].replace('"verdict"', "'verdict'")
        lines.append(f"DONE {len(s.fetched)}")
        return [M.TextPart("\n".join(lines))]

    def _verdict_line(self, series_id: str, values: list[float]) -> str:
        verdict = self._judge(series_id, values)
        return json.dumps(verdict, ensure_ascii=False)

    def _judge(self, series_id: str, values: list[float]) -> dict[str, Any]:
        """Fixed rule: a point beyond ``z_threshold`` robust sigmas, or a frozen run."""
        if len(values) < 3:
            return {"series_id": series_id, "verdict": "normal"}
        med = statistics.median(values)
        mad = statistics.median(abs(v - med) for v in values) * 1.4826
        scale = mad if mad > 0 else (statistics.pstdev(values) or 1.0)
        worst_i = max(range(len(values)), key=lambda i: abs(values[i] - med))
        worst_z = abs(values[worst_i] - med) / scale
        if worst_z > self.z_threshold:
            return {
                "series_id": series_id,
                "verdict": "anomalous",
                "explanation": (
                    f"value {values[worst_i]:.6g} at time_index {worst_i} lies {worst_z:.1f} robust sigmas "
                    f"from the median {med:.6g} (robust sigma {scale:.3g})"
                ),
            }
        run_start, run_len, best_start, best_len = 0, 1, 0, 1
        for i in range(1, len(values)):
            if values[i] == values[i - 1]:
                run_len += 1
            else:
                run_start, run_len = i, 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        if best_len >= 24 and mad > 0:
            return {
                "series_id": series_id,
                "verdict": "anomalous",
                "explanation": f"values are frozen at {values[best_start]:.6g} from time_index {best_start} to {best_start + best_len - 1} ({best_len} points)",
            }
        return {"series_id": series_id, "verdict": "normal"}

    # --- history ----------------------------------------------------------------------

    @staticmethod
    def _read(messages: list[M.ModelMessage]) -> _State:
        ids: list[str] | None = None
        fetched: dict[str, list[float]] = {}
        failed: list[str] = []
        verdicted: set[str] = set()
        last_returns: list[M.ToolReturnPart] = []
        retry: M.RetryPromptPart | None = None
        closing = False
        n_responses = 0
        for msg in messages:
            if isinstance(msg, M.ModelRequest):
                last_returns = []
                retry = None
                closing = n_responses > 0 and any(isinstance(p, M.UserPromptPart) for p in msg.parts)
                for part in msg.parts:
                    if isinstance(part, M.ToolReturnPart):
                        last_returns.append(part)
                        text = part.content if isinstance(part.content, str) else json.dumps(part.content)
                        if part.tool_name == "get_pool_meta_data" and not is_error(text):
                            ids = list(json.loads(text)["series_ids"])
                        elif part.tool_name == "get_series":
                            if is_error(text):
                                failed.append(_requested_id(text))
                            else:
                                doc = json.loads(text)
                                fetched[doc["series_id"]] = [float(v) for v in doc["value"]]
                    elif isinstance(part, M.RetryPromptPart):
                        retry = part
            elif isinstance(msg, M.ModelResponse):
                n_responses += 1
                for part in msg.parts:
                    if isinstance(part, M.TextPart):
                        for cand in parse_verdicts(part.content):
                            if cand.record is not None:
                                verdicted.add(cand.record.series_id)
        return _State(ids, fetched, failed, verdicted, last_returns, retry, closing, n_responses)

    def _estimate(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)


def _series_id_of(part: M.ToolReturnPart) -> str | None:
    text = part.content if isinstance(part.content, str) else None
    if text is None or is_error(text):
        return None
    try:
        return str(json.loads(text)["series_id"])
    except (ValueError, KeyError, TypeError):
        return None


def _requested_id(error_text: str) -> str:
    # "Error: unknown series_id 'X'. ..." / "Error: series 'X' is malformed ..."
    if "'" in error_text:
        return error_text.split("'", 2)[1]
    return "?"


def _history_text(messages: list[M.ModelMessage]) -> str:
    """What a provider would receive: the history plus the instructions once."""
    chunks: list[str] = []
    instructions: str | None = None
    for msg in messages:
        if isinstance(msg, M.ModelRequest) and msg.instructions:
            instructions = msg.instructions
        for part in msg.parts:
            if isinstance(part, M.ToolReturnPart):
                chunks.append(part.model_response_str())
            elif isinstance(part, M.ToolCallPart):
                chunks.append(part.args_as_json_str())
            else:
                content = getattr(part, "content", "")
                chunks.append(content if isinstance(content, str) else json.dumps(content, default=str))
    if instructions:
        chunks.append(instructions)
    return "\n".join(chunks)


def _parts_text(parts: list[M.ModelResponsePart]) -> str:
    out: list[str] = []
    for p in parts:
        if isinstance(p, M.TextPart):
            out.append(p.content)
        elif isinstance(p, M.ToolCallPart):
            out.append(p.tool_name + p.args_as_json_str())
    return "\n".join(out)


def scripted_model(**options: Any) -> FunctionModel:
    """A ``FunctionModel`` driven by :class:`ScriptedPolicy`."""
    return FunctionModel(ScriptedPolicy(**options), model_name=MODEL_NAME)
