"""The pydantic-ai agent of spec 02: two tools, greedy decoding, a request cap, one check.

What the harness adds on top of the two pure tool functions in ``tools.py`` is
bookkeeping only: timing and a trace record around every call (FR6), the record of which
series were examined, the id aliasing of ``ids.py``, and an output validator that
refuses to end the episode while a retrieved series still has no verdict (FR4/FR5: "a
verdict for each series it examines"). The validator never asks for more coverage of the
pool - stopping early is the agent's own decision and is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from .ids import IdAliases, PlainIds
from .pool import Pool
from .prompt import SYSTEM_PROMPT
from .tools import get_pool_meta_data as _pool_meta_data
from .tools import get_series as _get_series
from .tools import unknown_id_error
from .trace import TraceRecorder
from .verdicts import ParsedVerdict, VerdictCollector

TOOL_NAMES: tuple[str, ...] = ("get_series", "get_pool_meta_data")
AGENT_NAME = "raw-retrieval-baseline"


@dataclass
class EpisodeDeps:
    """Everything a tool call or the output validator may touch during one episode."""

    pool: Pool
    recorder: TraceRecorder
    collector: VerdictCollector
    ids: PlainIds | IdAliases | None = None
    value_precision: int | None = None
    enforce_verdicts: bool = True
    refused_call_ids: set[str] = field(default_factory=set, repr=False)  # retrievals the budget projection did not admit
    _payload_chars: dict[str, int] = field(default_factory=dict, repr=False)
    _meta_chars: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.ids is None:
            self.ids = PlainIds(self.pool.series_ids)

    def payload_chars(self, series_id: str) -> int:
        """Size of the ``get_series`` text for a pool id, computed once per series."""
        if series_id not in self._payload_chars:
            self._payload_chars[series_id] = len(
                _get_series(self.pool, series_id, value_precision=self.value_precision, label=self.ids.display(series_id))
            )
        return self._payload_chars[series_id]

    def meta_chars(self) -> int:
        if self._meta_chars is None:
            self._meta_chars = len(_pool_meta_data(self.pool, series_ids=self.ids.served))
        return self._meta_chars


def not_admitted_error(series_id: str) -> str:
    """The answer to a retrieval the budget projection did not admit: the calls issued before
    it in the same turn were served, this one was not, and the model is told why."""
    return (
        f"Error: not retrieved - series '{series_id}' would not fit in the episode's remaining budget together with "
        "the series requested before it in this turn. Request fewer series per turn."
    )


def get_series(ctx: RunContext[EpisodeDeps], series_id: str, **ignored: Any) -> str:
    """Return the full raw series with the given id.

    The result is a JSON document with the fields series_id, length, time_index (integer
    positions starting at 0) and value (one number per position). An unknown or malformed
    series_id yields an error message instead of data.

    Args:
        series_id: Identifier of the series, exactly as listed by get_pool_meta_data.
    """
    # ``**ignored``: stray arguments some models add (gpt-oss sends ``{"": ""}`` to a
    # no-argument tool) are dropped rather than refused - a refusal costs a whole
    # request on a rate-limited tier. The trace still records what the model sent.
    deps = ctx.deps
    real = deps.ids.resolve(series_id)
    with deps.recorder.tool_call(
        "get_series", ctx.tool_call_id, {"series_id": series_id, **ignored}, request_index=ctx.run_step, series_id=real if real is not None else series_id
    ) as call:
        if ctx.tool_call_id in deps.refused_call_ids:
            text = not_admitted_error(series_id)
            call.finish(text)
        elif real is None:
            # Only names the model was given resolve; with aliasing on, a pool id it may
            # have guessed from elsewhere is not one of them and the pool is never consulted.
            text = unknown_id_error(series_id) if series_id.strip() else _get_series(deps.pool, series_id)
        else:
            text = _get_series(deps.pool, real, value_precision=deps.value_precision, label=deps.ids.display(real))
        call.finish(text, n_points=deps.pool.length_of(real) if real is not None else None)
    if call.ok and real is not None:
        deps.collector.mark_examined(real)
    return text


def get_pool_meta_data(ctx: RunContext[EpisodeDeps], **ignored: Any) -> str:
    """Return the pool's metadata: its name, its size and the list of all series ids."""
    deps = ctx.deps
    with deps.recorder.tool_call("get_pool_meta_data", ctx.tool_call_id, dict(ignored), request_index=ctx.run_step) as call:
        text = _pool_meta_data(deps.pool, series_ids=deps.ids.served)
        call.finish(text)
    return text


def build_agent(
    model: Model | str,
    *,
    model_settings: ModelSettings | None = None,
    instructions: str = SYSTEM_PROMPT,
    tool_retries: int = 1,
    output_retries: int = 2,
    name: str = AGENT_NAME,
) -> Agent[EpisodeDeps, str]:
    """One agent, exactly two function tools, plain-text output.

    ``model_settings`` is where greedy decoding lives (``temperature=0``, ``seed``);
    the request cap is a run-time ``UsageLimits`` (see ``runner.py``), not an agent
    attribute. ``output_type`` stays ``str`` on purpose: an output tool would be a third
    tool exposed to the model, which FR3 forbids, so verdicts ride on the text instead.
    """
    agent: Agent[EpisodeDeps, str] = Agent(
        model,
        deps_type=EpisodeDeps,
        output_type=str,
        instructions=instructions,
        # sequential: each call runs alone, in the order the model issued them, so the
        # trace and the examination order never depend on thread timing (FR7).
        tools=[Tool(get_series, takes_ctx=True, sequential=True), Tool(get_pool_meta_data, takes_ctx=True, sequential=True)],
        model_settings=model_settings,
        retries={"tools": tool_retries, "output": output_retries},
        name=name,
    )

    @agent.output_validator
    def _every_examined_series_has_a_verdict(ctx: RunContext[EpisodeDeps], output: str) -> str:
        deps = ctx.deps
        # The runner ingests each response as it arrives; this is the fallback for a plain
        # ``agent.run`` and a no-op otherwise (ingestion is idempotent per request).
        deps.collector.ingest(output, ctx.run_step, deps.pool, resolve=deps.ids.resolve)
        if not deps.enforce_verdicts:
            return output
        missing = deps.collector.missing_verdicts()
        if not missing:
            return output
        last = deps.collector.results.get(ctx.run_step)
        raise ModelRetry(retry_message([deps.ids.display(s) for s in missing], last.rejected if last else []))

    return agent


def retry_message(missing: list[str], rejected: list[tuple[ParsedVerdict, str]], *, max_ids: int = 20) -> str:
    shown = ", ".join(missing[:max_ids])
    more = f" (and {len(missing) - max_ids} more)" if len(missing) > max_ids else ""
    text = (
        f"Your reply would end the episode, but {len(missing)} retrieved series still have no verdict: "
        f"{shown}{more}. Reply with one verdict line per series, exactly "
        '{"series_id": "<id>", "verdict": "normal"} or '
        '{"series_id": "<id>", "verdict": "anomalous", "explanation": "<evidence>"}, '
        "then the DONE line."
    )
    if rejected:
        notes = "; ".join(f"{cand.series_id} ({reason})" for cand, reason in rejected[:max_ids])
        text += f" Verdicts in your last message that were discarded: {notes}."
    return text


def registered_tool_names(agent: Agent[Any, Any]) -> list[str]:
    """Names of every function tool registered with ``agent`` (output tools excluded)."""
    names: list[str] = []
    for toolset in agent.toolsets:
        tools = getattr(toolset, "tools", None)
        if isinstance(tools, dict):
            names.extend(tools.keys())
    return names
