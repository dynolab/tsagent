"""One episode, end to end: agent loop, verdict capture, trace, output files (spec 02, §5).

``run_episode`` drives the pydantic-ai graph step by step (``agent.iter``) so that every
model round trip can be timed and its usage recorded, and so that the verdict lines in each
response are captured the moment they arrive - not only at the end. Tool calls run one at
a time in the order the model issued them, which keeps the trace replayable when a model
batches several ``get_series`` calls into one turn.

Whatever ends the episode - the agent stopping, the harness projecting that the next
request would break the token budget, a pydantic-ai usage limit, a provider error, Ctrl-C -
the outputs are closed properly: ``anomaly_verdicts.jsonl`` holds the standing verdicts,
``decision_trace.jsonl`` ends with a ``run_end`` record naming the termination reason,
``messages.json`` holds the conversation as pydantic-ai kept it, ``summary.json`` the
headline numbers.

``compare_runs`` is the FR7 check: two output directories are equal field-for-field once
wall-clock timing, provider-generated ids and cache accounting are set aside.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pydantic_ai
from pydantic_ai import Agent, UsageLimits
from pydantic_ai import messages as M
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from .agent import TOOL_NAMES, EpisodeDeps, build_agent, registered_tool_names
from .ids import IdAliases, PlainIds
from .models import SCRIPTED, build_model
from .pool import Pool
from .prompt import SYSTEM_PROMPT, USER_PROMPT, prompt_fingerprint
from .schemas import (
    LimitsInfo,
    ModelInfo,
    ModelRequestRecord,
    PoolInfo,
    PromptInfo,
    RetriesInfo,
    RetryKind,
    RunEndRecord,
    RunStartRecord,
    Termination,
    UsageFields,
    read_jsonl,
    strip_timing,
)
from .trace import TraceRecorder, now_iso
from .verdicts import VerdictCollector
from .writers import JsonlWriter, VerdictWriter, write_json

logger = logging.getLogger(__name__)

VERDICTS_FILE = "anomaly_verdicts.jsonl"
TRACE_FILE = "decision_trace.jsonl"
MESSAGES_FILE = "messages.json"
SUMMARY_FILE = "summary.json"
ALIASES_FILE = "series_aliases.json"

# Fields that legitimately differ between two replays of the same configuration: generated
# ids, wall-clock stamps, and the provider's cache accounting (a warm prompt cache changes
# how tokens are billed, not what the model saw or said).
REPLAY_VARIANT_TRACE_FIELDS = frozenset({"tool_call_id", "cache_read_tokens", "cache_write_tokens", "details"})
REPLAY_VARIANT_MESSAGE_FIELDS = frozenset(
    {
        "timestamp",
        "tool_call_id",
        "run_id",
        "conversation_id",
        "provider_response_id",
        "provider_details",
        "provider_name",
        "provider_url",
        "id",
        "cache_read_tokens",
        "cache_write_tokens",
        "details",
    }
)

TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

NOT_EXECUTED = "Error: not retrieved - the episode's token budget is exhausted. No further tool call will be executed."


@dataclass(frozen=True)
class EpisodeConfig:
    """Everything that determines an episode's behaviour besides the pool itself."""

    model: str
    temperature: float | None = 0.0
    seed: int | None = 42
    max_tokens: int | None = None
    parallel_tool_calls: bool | None = False
    extra_model_settings: dict[str, Any] = field(default_factory=dict)
    order_seed: int | None = None  # None: serve the pool in file order
    alias_series_ids: bool = True  # opaque ids for the model; see ids.py
    request_limit: int | None = None  # None: 2 requests per series + 8
    tool_calls_limit: int | None = None  # None: 2 calls per series + 8
    total_tokens_limit: int | None = 1_000_000
    context_tokens_limit: int | None = None  # None: rely on the provider's own window
    chars_per_token_estimate: float = 1.5  # conservative for numeric JSON (measured 1.6-2.2)
    output_allowance_tokens: int = 1024  # expected response size when max_tokens is unset
    tool_retries: int = 1
    output_retries: int = 2
    enforce_verdicts: bool = True
    value_precision: int | None = None
    system_prompt: str = SYSTEM_PROMPT
    user_prompt: str = USER_PROMPT
    # Deployment, not behaviour - excluded from the fingerprint.
    provider_base_url: str | None = None
    provider_anonymous: bool = False  # strip the Authorization header (keyless endpoints)
    provider_timeout_s: float = 120.0
    provider_max_retries: int = 0  # the OpenAI SDK's own quick retries; the harness retries below
    request_interval_s: float = 0.0  # pacing between model requests, for rate-limited tiers
    transient_retries: int = 8  # harness retries on HTTP 429 / 5xx, waiting for the provider's reset
    rate_limit_wait_s: float = 35.0  # wait after a 429 when the provider names no reset time
    scripted_options: dict[str, Any] = field(default_factory=dict)

    DEPLOYMENT_FIELDS = (
        "provider_base_url",
        "provider_anonymous",
        "provider_timeout_s",
        "provider_max_retries",
        "request_interval_s",
        "transient_retries",
        "rate_limit_wait_s",
    )

    def model_settings(self) -> ModelSettings | None:
        settings: dict[str, Any] = {}
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        if self.seed is not None:
            settings["seed"] = self.seed
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        if self.parallel_tool_calls is not None:
            settings["parallel_tool_calls"] = self.parallel_tool_calls
        settings.update(self.extra_model_settings)
        return ModelSettings(**settings) if settings else None  # type: ignore[typeddict-item]

    def effective_request_limit(self, n_series: int) -> int | None:
        return self.request_limit if self.request_limit is not None else 2 * n_series + 8

    def effective_tool_calls_limit(self, n_series: int) -> int | None:
        return self.tool_calls_limit if self.tool_calls_limit is not None else 2 * n_series + 8

    def usage_limits(self, n_series: int) -> UsageLimits:
        return UsageLimits(
            request_limit=self.effective_request_limit(n_series),
            tool_calls_limit=self.effective_tool_calls_limit(n_series),
            total_tokens_limit=self.total_tokens_limit,
        )

    def limits_info(self, n_series: int) -> LimitsInfo:
        return LimitsInfo(
            request_limit=self.effective_request_limit(n_series),
            tool_calls_limit=self.effective_tool_calls_limit(n_series),
            total_tokens_limit=self.total_tokens_limit,
            context_tokens_limit=self.context_tokens_limit,
            chars_per_token_estimate=float(self.chars_per_token_estimate),
            output_allowance_tokens=self.output_allowance_tokens,
        )

    def fingerprint(self) -> str:
        """Identity of the configuration - excludes the endpoint address, which is deployment, not behaviour."""
        payload = asdict(self)
        for name in self.DEPLOYMENT_FIELDS:
            payload.pop(name)
        if self.model != SCRIPTED:
            payload.pop("scripted_options")  # only the stand-in reads them
        payload["system_prompt"] = prompt_fingerprint(self.system_prompt)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class EpisodeResult:
    output_dir: Path
    run_id: str
    termination: Termination
    run_end: RunEndRecord
    files: dict[str, Path]

    @property
    def usage(self) -> UsageFields:
        return self.run_end.usage


class _RecordingModel(WrapperModel):
    """Two jobs around the real model.

    It keeps every response the model returned, so one that pydantic-ai drops after a
    usage limit trips can still be booked exactly (usage, details, verdict text). And it
    absorbs transient HTTP failures - a 429 from a rate-limited tier, a 5xx - by waiting
    and asking again, with the wait taken from the provider's reset headers when it sends
    them, so the episode sees one slow request instead of a ``model_error``.
    """

    def __init__(self, wrapped: Model, *, transient_retries: int = 0, rate_limit_wait_s: float = 35.0, server_error_wait_s: float = 10.0):
        super().__init__(wrapped)
        self.last_response: M.ModelResponse | None = None
        self.responses = 0
        self.transient_retries = transient_retries
        self.rate_limit_wait_s = rate_limit_wait_s
        self.server_error_wait_s = server_error_wait_s
        self.transient_failures = 0

    async def request(
        self, messages: list[M.ModelMessage], model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> M.ModelResponse:
        attempt = 0
        while True:
            try:
                response = await super().request(messages, model_settings, model_request_parameters)
                break
            except ModelHTTPError as exc:
                if exc.status_code not in TRANSIENT_HTTP_STATUSES or attempt >= self.transient_retries:
                    raise
                attempt += 1
                self.transient_failures += 1
                wait = _retry_wait_seconds(exc, default=self.rate_limit_wait_s if exc.status_code == 429 else self.server_error_wait_s)
                logger.warning("HTTP %d from the model endpoint; waiting %.0fs before retry %d/%d", exc.status_code, wait, attempt, self.transient_retries)
                await asyncio.sleep(wait)
        self.last_response = response
        self.responses += 1
        return response


def _retry_wait_seconds(exc: ModelHTTPError, *, default: float) -> float:
    """The provider's own advice on when to come back, else ``default``; never less than 1 s."""
    headers = {str(k).lower(): str(v) for k, v in (getattr(exc, "headers", None) or {}).items()}
    for key in ("retry-after", "ratelimit-reset", "x-ratelimit-reset"):
        value = headers.get(key)
        if value is None:
            continue
        try:
            return max(1.0, float(value) + 2.0)
        except ValueError:
            continue
    return max(1.0, default)


# --- the episode ------------------------------------------------------------------------


def run_episode(
    pool: Pool,
    config: EpisodeConfig,
    output_dir: str | Path,
    *,
    model: Model | None = None,
    api_key: str | None = None,
    overwrite: bool = False,
) -> EpisodeResult:
    """Run one episode over ``pool`` and write its outputs into ``output_dir``."""
    return asyncio.run(run_episode_async(pool, config, output_dir, model=model, api_key=api_key, overwrite=overwrite))


async def run_episode_async(
    pool: Pool,
    config: EpisodeConfig,
    output_dir: str | Path,
    *,
    model: Model | None = None,
    api_key: str | None = None,
    overwrite: bool = False,
) -> EpisodeResult:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir} (pass overwrite=True to reuse it)")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {name: output_dir / name for name in (VERDICTS_FILE, TRACE_FILE, MESSAGES_FILE, SUMMARY_FILE)}

    if model is None:
        model = build_model(
            config.model,
            base_url=config.provider_base_url,
            api_key=api_key,
            anonymous=config.provider_anonymous,
            timeout_s=config.provider_timeout_s,
            max_retries=config.provider_max_retries,
            scripted_options=config.scripted_options,
        )
    recording = _RecordingModel(model, transient_retries=config.transient_retries, rate_limit_wait_s=config.rate_limit_wait_s)
    agent = build_agent(
        recording,
        model_settings=config.model_settings(),
        instructions=config.system_prompt,
        tool_retries=config.tool_retries,
        output_retries=config.output_retries,
    )
    tools = registered_tool_names(agent)
    if sorted(tools) != sorted(TOOL_NAMES):
        raise RuntimeError(f"agent exposes tools {tools}, expected exactly {list(TOOL_NAMES)}")

    served = pool.reordered(config.order_seed)
    ids = IdAliases(served.series_ids) if config.alias_series_ids else PlainIds(served.series_ids)
    run_id = hashlib.sha256((pool.fingerprint + config.fingerprint()).encode()).hexdigest()[:12]
    collector = VerdictCollector()
    verdict_writer = VerdictWriter(files[VERDICTS_FILE])
    trace_writer = JsonlWriter(files[TRACE_FILE])
    recorder = TraceRecorder(trace_writer, served)
    deps = EpisodeDeps(
        pool=served,
        recorder=recorder,
        collector=collector,
        ids=ids,
        value_precision=config.value_precision,
        enforce_verdicts=config.enforce_verdicts,
    )
    limits = config.usage_limits(len(served))

    recorder.start(
        RunStartRecord(
            run_id=run_id,
            config_fingerprint=config.fingerprint(),
            pool=PoolInfo(name=pool.name, path=pool.source, n_series=len(pool), fingerprint=pool.fingerprint),
            order_seed=config.order_seed,
            series_id_aliasing=config.alias_series_ids,
            model=ModelInfo(name=config.model, settings=dict(config.model_settings() or {}), provider_base_url=config.provider_base_url),
            usage_limits=config.limits_info(len(served)),
            retries=RetriesInfo(tools=config.tool_retries, output=config.output_retries),
            prompt=PromptInfo(system_prompt_sha256=prompt_fingerprint(config.system_prompt), user_prompt=config.user_prompt),
            tools=tools,
            value_precision=config.value_precision,
            harness={"pydantic_ai": pydantic_ai.__version__, "python": platform.python_version(), "platform": platform.platform()},
            started_at=now_iso(),
        )
    )
    if ids.aliased:
        files[ALIASES_FILE] = output_dir / ALIASES_FILE
        write_json(files[ALIASES_FILE], ids.mapping())
    logger.info(
        "run %s: pool %s (%d series, order seed %s, aliases %s), model %s, limits: %s requests, %s tool calls, %s tokens",
        run_id,
        pool.name,
        len(served),
        config.order_seed,
        "on" if ids.aliased else "off",
        config.model,
        limits.request_limit,
        limits.tool_calls_limit,
        limits.total_tokens_limit,
    )

    try:
        termination, messages = await _drive(agent, recording, deps, config, limits, verdict_writer)
    except BaseException:
        # Anything _drive did not translate into a termination (SystemExit, a second
        # Ctrl-C) still leaves a closed trace behind.
        recorder.finish(Termination(reason="interrupted", detail="aborted before completion"), collector)
        trace_writer.close()
        raise
    run_end = recorder.finish(termination, collector)
    trace_writer.close()
    write_json(files[MESSAGES_FILE], ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
    summary = {
        "run_id": run_id,
        "pool": pool.name,
        "model": config.model,
        "output_dir": str(output_dir),
        **run_end.model_dump(mode="json"),
    }
    write_json(files[SUMMARY_FILE], summary)
    logger.info(
        "run %s finished: %s; %d requests, %d get_series calls, %d/%d series verdicted (%d anomalous), %d tokens",
        run_id,
        termination.reason,
        run_end.requests,
        run_end.tool_calls.get_series,
        run_end.series.verdicted,
        len(served),
        run_end.series.anomalous,
        run_end.usage.total_tokens,
    )
    return EpisodeResult(output_dir=output_dir, run_id=run_id, termination=termination, run_end=run_end, files=files)


async def _drive(
    agent: Agent[EpisodeDeps, str],
    recording: _RecordingModel,
    deps: EpisodeDeps,
    config: EpisodeConfig,
    limits: UsageLimits,
    verdict_writer: VerdictWriter,
) -> tuple[Termination, list[M.ModelMessage]]:
    """Step through the agent graph, recording each model request and ingesting its text.

    When the budget projection stops the episode before a retrieval, and the model still
    owes verdicts for series it did retrieve, one closing turn follows: the pending call is
    answered with :data:`NOT_EXECUTED` and the model is asked for the verdicts it owes -
    the same rule the output validator applies when the model stops by itself, never a
    request to examine anything more.
    """
    recorder, collector, pool = deps.recorder, deps.collector, deps.pool
    request_index = 0
    termination = Termination(reason="completed")
    run: Any = None
    pending: dict[str, Any] | None = None  # timing of the request in flight
    last_request_at: float | None = None

    def _record_response(response: M.ModelResponse, *, started_at: str, duration_ms: float, retry_kind: RetryKind | None, history_chars: int) -> ModelRequestRecord:
        text = "\n".join(p.content for p in response.parts if isinstance(p, M.TextPart))
        tool_parts = [p for p in response.parts if isinstance(p, M.ToolCallPart)]
        ingest = collector.ingest(text, request_index, pool, resolve=deps.ids.resolve)
        for rec in ingest.accepted:
            verdict_writer.put(rec)
        requested = [str(_args_as_dict(p).get("series_id")) for p in tool_parts if p.tool_name == "get_series"]
        record = recorder.model_request(
            request_index=request_index,
            retry_kind=retry_kind,
            started_at=started_at,
            duration_ms=duration_ms,
            usage=response.usage,
            model_name=response.model_name,
            finish_reason=response.finish_reason,
            history_chars=history_chars,
            response_text_chars=len(text),
            n_tool_calls=len(tool_parts),
            requested_series=requested,
            verdicts_accepted=len(ingest.accepted),
            verdicts_rejected=len(ingest.rejected),
        )
        recorder.verdicts(ingest.events)
        logger.info(
            "request %d: in=%s out=%s tokens, context=%d chars, %d tool call(s) %s, verdicts +%d/-%d (%d standing)",
            request_index,
            response.usage.input_tokens,
            response.usage.output_tokens,
            history_chars,
            len(tool_parts),
            requested[:3],
            len(ingest.accepted),
            len(ingest.rejected),
            len(collector.final),
        )
        return record

    async def _loop(active_run: Any, *, execute_tools: bool, first_kind: RetryKind | None) -> tuple[Termination | None, M.ModelResponse | None]:
        """Advance ``active_run`` to its end. Returns a budget stop (and the response whose
        tool calls were not executed) or, with ``execute_tools`` off, the first response
        that asked for a tool."""
        nonlocal request_index, pending, last_request_at
        node = active_run.next_node
        first = True
        while not Agent.is_end_node(node):
            if Agent.is_model_request_node(node):
                request_index += 1
                history = list(active_run.all_messages())
                if not history or history[-1] is not node.request:
                    history.append(node.request)
                if config.request_interval_s > 0 and last_request_at is not None:
                    wait = config.request_interval_s - (time.perf_counter() - last_request_at)
                    if wait > 0:
                        await asyncio.sleep(wait)
                last_request_at = time.perf_counter()
                pending = {
                    "retry_kind": _retry_kind(node.request.parts) or (first_kind if first else None),
                    "history_chars": history_chars(history, instructions=config.system_prompt),
                    "started_at": now_iso(),
                    "t0": time.perf_counter(),
                }
                first = False
                next_node = await active_run.next(node)
                duration_ms = (time.perf_counter() - pending["t0"]) * 1000.0
                if Agent.is_call_tools_node(next_node):
                    response = next_node.model_response
                    record = _record_response(
                        response,
                        started_at=pending["started_at"],
                        duration_ms=duration_ms,
                        retry_kind=pending["retry_kind"],
                        history_chars=pending["history_chars"],
                    )
                    pending = None
                    has_tool_calls = any(isinstance(p, M.ToolCallPart) for p in response.parts)
                    if has_tool_calls and not execute_tools:
                        return None, response
                    stop, refused = _admit_calls(record, response, deps, config)
                    if stop is not None:
                        return stop, response
                    if refused:
                        deps.refused_call_ids.update(p.tool_call_id for p in refused)
                        asked = sum(1 for p in response.parts if isinstance(p, M.ToolCallPart) and p.tool_name == "get_series")
                        logger.info("request %d asked for %d retrievals; %d admitted, %d refused (they would not fit)", request_index, asked, asked - len(refused), len(refused))
                pending = None
                node = next_node
            elif Agent.is_call_tools_node(node):
                async with node.stream(active_run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, M.FunctionToolResultEvent) and isinstance(event.part, M.RetryPromptPart):
                            recorder.invalid_tool_call(
                                event.part.tool_name or "?",
                                event.part.tool_call_id,
                                _args_of(node.model_response, event.part.tool_call_id),
                                event.part.model_response(),
                                request_index=request_index,
                            )
                node = await active_run.next(node)
            else:
                node = await active_run.next(node)
        return None, None

    try:
        with Agent.parallel_tool_call_execution_mode("sequential"):
            async with agent.iter(config.user_prompt, deps=deps, usage_limits=limits) as run:
                stop, stopped_response = await _loop(run, execute_tools=True, first_kind=None)
            if stop is not None:
                termination = stop
                if stopped_response is not None:
                    termination, run = await _closing_turn(agent, run, stopped_response, termination, deps, config, limits, _loop)
    except UsageLimitExceeded as exc:
        termination = Termination(reason="usage_limit", detail=_short(exc))
    except UnexpectedModelBehavior as exc:
        termination = Termination(reason="protocol_error", detail=_short(exc))
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        # Under asyncio.run a Ctrl-C arrives as a cancellation of this task.
        termination = Termination(reason="interrupted", detail=type(exc).__name__)
    except Exception as exc:  # provider/API failures: recorded, not swallowed silently
        logger.exception("episode aborted during request %d", request_index)
        termination = Termination(reason="model_error", detail=_short(exc))

    messages = list(run.all_messages()) if run is not None else []
    if pending is not None and recording.responses > len(recorder.requests) and recording.last_response is not None:
        # A limit that trips *after* a response arrives (total_tokens_limit) makes
        # pydantic-ai drop that response before it reaches the history. The wrapper kept
        # it, so it is booked like any other - usage, details and verdict lines included.
        _record_response(
            recording.last_response,
            started_at=pending["started_at"],
            duration_ms=(time.perf_counter() - pending["t0"]) * 1000.0,
            retry_kind=pending["retry_kind"],
            history_chars=pending["history_chars"],
        )
        if not any(m is recording.last_response for m in messages):
            messages.append(recording.last_response)
    return termination, messages


async def _closing_turn(
    agent: Agent[EpisodeDeps, str],
    run: Any,
    stopped_response: M.ModelResponse,
    termination: Termination,
    deps: EpisodeDeps,
    config: EpisodeConfig,
    limits: UsageLimits,
    loop: Any,
) -> tuple[Termination, Any]:
    """After a budget stop: collect the verdicts owed for retrieved series, if that fits."""
    recorder, collector = deps.recorder, deps.collector
    owed = collector.missing_verdicts()
    if not owed or not deps.enforce_verdicts:
        return termination, run
    last = recorder.requests[-1]
    cpt = config.chars_per_token_estimate
    context = last.usage.input_tokens if last.usage.input_tokens > 0 else math.ceil(last.history_chars / cpt)
    allowance = config.max_tokens if config.max_tokens is not None else config.output_allowance_tokens
    closing_input = context + last.usage.output_tokens + 400
    over_context = config.context_tokens_limit is not None and closing_input + allowance > config.context_tokens_limit
    over_budget = config.total_tokens_limit is not None and recorder.total_tokens() + closing_input + allowance > config.total_tokens_limit
    if over_context or over_budget:
        note = f" [no closing turn: it would not fit either; {len(owed)} retrieved series remain unjudged]"
        return Termination(reason=termination.reason, detail=(termination.detail or "") + note), run
    pending_calls = [p for p in stopped_response.parts if isinstance(p, M.ToolCallPart)]
    for part in pending_calls:
        args = _args_as_dict(part)
        sid = args.get("series_id")
        real = deps.ids.resolve(sid) if isinstance(sid, str) else None
        recorder.skipped_tool_call(part.tool_name, part.tool_call_id, args, request_index=last.request_index, series_id=real if real is not None else (sid if isinstance(sid, str) else None))
    returns = [M.ToolReturnPart(tool_name=p.tool_name, content=NOT_EXECUTED, tool_call_id=p.tool_call_id) for p in pending_calls]
    history = list(run.all_messages())
    history.append(M.ModelRequest(parts=[*returns, M.UserPromptPart(closing_prompt([deps.ids.display(s) for s in owed]))]))
    logger.info("budget stop with %d retrieved series unjudged; asking for their verdicts in a closing turn", len(owed))
    async with agent.iter(None, message_history=history, deps=deps, usage_limits=limits, usage=run.usage) as closing_run:
        _, refused = await loop(closing_run, execute_tools=False, first_kind="closing")
    still_owed = collector.missing_verdicts()
    if refused is not None:
        note = f" [closing turn asked for another tool call instead of verdicts; not executed; {len(still_owed)} retrieved series remain unjudged]"
    elif still_owed:
        note = f" [closing turn left {len(still_owed)} retrieved series unjudged]"
    else:
        note = " [closing turn collected the verdicts for the retrieved series]"
    return Termination(reason=termination.reason, detail=(termination.detail or "") + note), closing_run


def closing_prompt(owed_ids: Sequence[str], *, max_ids: int = 20) -> str:
    shown = ", ".join(owed_ids[:max_ids])
    more = f" (and {len(owed_ids) - max_ids} more)" if len(owed_ids) > max_ids else ""
    return (
        "The episode's token budget is exhausted: the retrieval you just requested was not performed, and no "
        f"further tool call will be executed. Reply now with one verdict line for each series you retrieved that "
        f"still has no verdict - {shown}{more} - then the DONE line."
    )


# --- budget projection ------------------------------------------------------------------


def _admit_calls(record: ModelRequestRecord, response: M.ModelResponse, deps: EpisodeDeps, config: EpisodeConfig) -> tuple[Termination | None, list[M.ToolCallPart]]:
    """Decide, before any tool of ``response`` runs, which of its retrievals the request that
    would follow can afford. pydantic-ai checks token limits after a response has been paid
    for - and the request after a retrieval is the expensive one - so the projection here
    is what keeps a capped run under its cap and free of retrieved-but-unjudged series.

    Retrievals are admitted in issue order while the projection fits; the rest of the turn
    is refused, and the tool answers each refused call with ``not_admitted_error`` so the
    model reads why. A turn whose first retrieval does not fit is a budget stop."""
    tool_parts = [p for p in response.parts if isinstance(p, M.ToolCallPart)]
    if not tool_parts or (config.total_tokens_limit is None and config.context_tokens_limit is None):
        return None, []
    cpt = config.chars_per_token_estimate
    context = record.usage.input_tokens if record.usage.input_tokens > 0 else math.ceil(record.history_chars / cpt)
    allowance = config.max_tokens if config.max_tokens is not None else config.output_allowance_tokens
    spent = deps.recorder.total_tokens()

    def over(payload_chars: int) -> str | None:
        payload_tokens = math.ceil(payload_chars / cpt)
        next_input = context + record.usage.output_tokens + payload_tokens
        breakdown = f"context {context:,} + response {record.usage.output_tokens:,} + requested payloads ~{payload_tokens:,}"
        if config.context_tokens_limit is not None and next_input + allowance > config.context_tokens_limit:
            return (
                f"the next request would need about {next_input:,} input tokens ({breakdown}) plus {allowance:,} for the reply, "
                f"over context_tokens_limit={config.context_tokens_limit:,}"
            )
        projected = spent + next_input + allowance
        if config.total_tokens_limit is not None and projected > config.total_tokens_limit:
            return (
                f"the next request would bring the episode to about {projected:,} tokens ({spent:,} spent + {next_input:,} input [{breakdown}] "
                f"+ {allowance:,} reply allowance), over total_tokens_limit={config.total_tokens_limit:,}"
            )
        return None

    fixed = 0  # payloads of the calls that are always served: metadata, and anything that is not a retrieval
    retrievals: list[tuple[M.ToolCallPart, str, int]] = []
    for part in tool_parts:
        if part.tool_name == "get_series":
            sid = _args_as_dict(part).get("series_id")
            real = deps.ids.resolve(sid) if isinstance(sid, str) else None
            retrievals.append((part, str(sid), deps.payload_chars(real) if real is not None else 160))
        elif part.tool_name == "get_pool_meta_data":
            fixed += deps.meta_chars()
        else:
            fixed += 160
    if not retrievals:
        reason = over(fixed)
        return (Termination(reason="budget", detail=f"{reason}; not fetched: []") if reason else None), []
    admitted = 0
    refused: list[M.ToolCallPart] = []
    for part, _sid, chars in retrievals:
        if refused or over(fixed + admitted + chars) is not None:
            refused.append(part)
        else:
            admitted += chars
    if len(refused) == len(retrievals):
        reason = over(fixed + sum(chars for _, _, chars in retrievals))
        return Termination(reason="budget", detail=f"{reason}; not fetched: {[sid for _, sid, _ in retrievals]}"), []
    return None, refused

# --- helpers ----------------------------------------------------------------------------


def history_chars(messages: Sequence[M.ModelMessage], *, instructions: str | None = None) -> int:
    """Size in characters of everything the model is sent - a provider-independent context measure.

    pydantic-ai stamps the instructions on every request it keeps in the history but sends
    them once, so they are counted once: the latest copy in the history, or ``instructions``
    when none is there yet (before the first request is prepared).
    """
    total = 0
    carried: str | None = None
    for msg in messages:
        if isinstance(msg, M.ModelRequest) and msg.instructions:
            carried = msg.instructions
        for part in msg.parts:
            if isinstance(part, M.ToolReturnPart):
                total += len(part.model_response_str())
            elif isinstance(part, M.ToolCallPart):
                total += len(part.tool_name) + len(part.args_as_json_str())
            elif isinstance(part, M.RetryPromptPart):
                total += len(part.model_response())
            else:
                content = getattr(part, "content", "")
                total += len(content) if isinstance(content, str) else len(json.dumps(content, default=str))
    total += len((carried if carried is not None else (instructions or "")).strip())
    return total


def _retry_kind(parts: Sequence[Any]) -> RetryKind | None:
    retries = [p for p in parts if isinstance(p, M.RetryPromptPart)]
    if not retries:
        return None
    return "tool" if any(p.tool_name for p in retries) else "output"


def _args_as_dict(part: M.ToolCallPart) -> dict[str, Any]:
    try:
        args = part.args_as_dict()
    except Exception:
        return {"raw": part.args}
    return args if isinstance(args, dict) else {"raw": args}


def _args_of(response: M.ModelResponse, tool_call_id: str | None) -> Any:
    for part in response.parts:
        if isinstance(part, M.ToolCallPart) and part.tool_call_id == tool_call_id:
            return _args_as_dict(part)
    return {}


def _short(exc: BaseException, limit: int = 500) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --- FR7: replay comparison -------------------------------------------------------------


@dataclass
class RunDiff:
    identical: bool
    differences: list[str]


def _strip(obj: Any, fields: frozenset[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v, fields) for k, v in obj.items() if k not in fields}
    if isinstance(obj, list):
        return [_strip(v, fields) for v in obj]
    return obj


def _first_difference(a: list[Any], b: list[Any], label: str) -> str | None:
    if len(a) != len(b):
        return f"{label}: {len(a)} vs {len(b)} records"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            kind = x.get("kind", "?") if isinstance(x, dict) else "?"
            return f"{label}: record {i + 1} ({kind}) differs: {json.dumps(x, ensure_ascii=False)[:200]} != {json.dumps(y, ensure_ascii=False)[:200]}"
    return None


def compare_runs(dir_a: str | Path, dir_b: str | Path, *, compare_messages: bool = True) -> RunDiff:
    """Field-for-field equality of two episode outputs, ignoring timing, generated ids and cache accounting."""
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    diffs: list[str] = []
    d = _first_difference(read_jsonl(dir_a / VERDICTS_FILE), read_jsonl(dir_b / VERDICTS_FILE), VERDICTS_FILE)
    if d:
        diffs.append(d)
    trace_a = _strip(strip_timing(read_jsonl(dir_a / TRACE_FILE)), REPLAY_VARIANT_TRACE_FIELDS)
    trace_b = _strip(strip_timing(read_jsonl(dir_b / TRACE_FILE)), REPLAY_VARIANT_TRACE_FIELDS)
    d = _first_difference(trace_a, trace_b, TRACE_FILE)
    if d:
        diffs.append(d)
    if compare_messages:
        msgs_a = _strip(json.loads((dir_a / MESSAGES_FILE).read_text(encoding="utf-8")), REPLAY_VARIANT_MESSAGE_FIELDS)
        msgs_b = _strip(json.loads((dir_b / MESSAGES_FILE).read_text(encoding="utf-8")), REPLAY_VARIANT_MESSAGE_FIELDS)
        d = _first_difference(msgs_a, msgs_b, MESSAGES_FILE)
        if d:
            diffs.append(d)
    return RunDiff(identical=not diffs, differences=diffs)
