# 02 - Raw Retrieval Baseline Agent Infrastructure

The raw-retrieval-only baseline agent of RQ1: one pydantic-ai agent with exactly two tools,
`get_series` and `get_pool_meta_data`, that works through a pool in a single episode and
emits a verdict for every series it examines. It is the reference point the tool-augmented
agent of M2/M3 is compared against, and the reference point of RQ3's context-efficiency
measurements, so its token consumption is logged down to the request.

The contract is
[`.internal/specs/02-raw-retrieval-baseline-agent-infrastructure-spec.md`](../../.internal/specs/02-raw-retrieval-baseline-agent-infrastructure-spec.md)
(issue [dynolab/tsagent#2](https://github.com/dynolab/tsagent/issues/2)). As the spec
asks, the code lives in [`code/`](../../code/), not here; this file covers what is done,
how to run it and what to expect from it.

## Status

- [x] Pool storage and the two tools (FR1, FR2) -
      [`pool.py`](../../code/src/baseline_agent/pool.py),
      [`tools.py`](../../code/src/baseline_agent/tools.py). The `label` column is never
      read; unknown or malformed ids come back as an error sentence, not an exception.
- [x] The agent (FR3, spec §4.2): two function tools and nothing else, plain-text output,
      greedy decoding, one tool call at a time, the B1 budget prompt with the
      stop-when-confident rule - [`agent.py`](../../code/src/baseline_agent/agent.py),
      [`prompt.py`](../../code/src/baseline_agent/prompt.py).
- [x] One episode per pool with the agent choosing what to examine (FR4), the verdict
      protocol and writer (FR5), the decision trace with per-call timing and per-request
      token usage (FR6, NFR) -
      [`runner.py`](../../code/src/baseline_agent/runner.py),
      [`verdicts.py`](../../code/src/baseline_agent/verdicts.py),
      [`trace.py`](../../code/src/baseline_agent/trace.py),
      [`writers.py`](../../code/src/baseline_agent/writers.py).
- [x] Entry point [`run_baseline_agent.py`](../../code/src/run_baseline_agent.py) with
      output validation and the replay comparison behind FR7 (`REPLAYS > 1`).
- [x] Unit tests, no LLM and no GPU (spec §3, unit level): tool registration and the
      settings the model receives, tool behaviour, strict output schemas, and whole episodes
      driven by a scripted stand-in model through every termination path -
      [`code/tests/`](../../code/tests/), 101 tests.
- [x] Dry runs on the real R1 pool with the stand-in: pipeline, trace consistency and
      replay identity hold on real data; [`src/measure_payloads.py`](src/measure_payloads.py)
      reproduces the context arithmetic below.
- [x] End-to-end runs with an LLM (spec §3, E2E level) on two free endpoints, gpt-oss-20b
      through OVHcloud's anonymous tier and GLM-4.7-Flash through Z.AI's free tier: replay
      pairs and full R1 and R2 runs, with opaque ids and, before the switch, with the
      pool's own - see [End-to-end runs on free endpoints](#end-to-end-runs-on-free-endpoints).
      The harness behaves as designed; neither endpoint replays deterministically; gpt-oss
      examines one to three series per episode, GLM fills the window with three to ten.
- [ ] The same checks on a locally served model with a fixed batch size, where the FR7
      replay can be verified end to end, and enough episodes per pool to characterise
      coverage beyond a single run.

## Layout

```
code/src/baseline_agent/
  pool.py        Pool: label-free view of a spec-01 parquet file, seeded reordering
  tools.py       get_series / get_pool_meta_data as pure functions returning text
  ids.py         the ids the model sees: opaque aliases by default, or the pool's own
  prompt.py      system prompt (task, verdict format, budget rule, stop rule), user prompt
  agent.py       pydantic-ai Agent: the two tools with timing/trace wrappers, output check
  verdicts.py    verdict-line parser and the episode rules (examined-only, last wins)
  trace.py       decision-trace recorder, per-series roll-ups, token attribution
  writers.py     JSONL writers (verdicts file keeps one record per series)
  runner.py      run_episode: agent.iter() loop, budget projection, termination handling,
                 compare_runs (FR7)
  models.py      model string -> pydantic-ai Model (providers, OpenAI-compatible endpoints)
  scripted.py    deterministic stand-in for the LLM - dry runs and tests only
code/src/run_baseline_agent.py   configuration-driven entry point
code/tests/                      test_tools.py, test_output_schemas.py, test_runner.py
src/measure_payloads.py          payload and token measurements behind the table below
src/summarize_run.py             coverage, cost, termination and label read-out of runs
results/<run>/                   verdicts, trace, summary and alias map of the real-model runs
(full runs, messages.json included, land in
 ../../data/02-raw-retrieval-baseline-agent-infrastructure/runs/, git-ignored)
```

## Running

Install once and make sure the pools are in place (`dvc pull`, or rebuild them with the
task-01 scripts):

```bash
pip install -r code/requirements.txt
```

The entry point follows the repository convention: every knob sits in the `CONFIGURATION`
block at the top of the file, there are no command-line arguments. Edit the block, run the
file:

```bash
python code/src/run_baseline_agent.py
```

The defaults make a **dry run**: `MODEL = "scripted"` replaces the LLM with a
deterministic stand-in that walks the served order of R1, retrieves each series once,
judges it with a fixed rule and moves on, until the budget projection stops the episode
(14 series with the default 1M-token cap; its token counts are estimates, see below). It
exercises everything except the model and needs no API. Its verdicts say nothing about
anomaly detection.

Useful knobs:

| Knob | Meaning |
|------|---------|
| `POOL_FILE`, `SPLIT`, `SERIES_LIMIT` | which pool, optionally one split (`dev`/`test`) or the first N series of the served order |
| `ORDER_SEED` | seeded permutation of the order the agent is shown (`None` = file order) |
| `ALIAS_SERIES_IDS` | on by default: the agent sees opaque ids instead of the provenance-carrying spec-01 ids; `False` serves the pool's ids as they are and marks the run name with `plainids` |
| `MODEL`, `MODEL_BASE_URL`, `MODEL_API_KEY_ENV` / `MODEL_API_KEY_FILE`, `MODEL_ANONYMOUS` | which LLM, where, and how to authenticate (see below) |
| `REQUEST_INTERVAL_S`, `TRANSIENT_RETRIES`, `RATE_LIMIT_WAIT_S` | pacing for rate-limited tiers, and how long the harness keeps retrying 429 / 5xx answers |
| `TEMPERATURE = 0.0`, `SEED = 42`, `PARALLEL_TOOL_CALLS = False` | greedy, seeded decoding, one retrieval per turn (FR7, idea B) |
| `REQUEST_LIMIT`, `TOOL_CALLS_LIMIT` | pydantic-ai's episode caps; both default to 2 per series + 8 |
| `TOTAL_TOKENS_LIMIT`, `CONTEXT_TOKENS_LIMIT` | the token budget (default: the constitution's 1M) and the model's window; the harness stops *before* the request that would break either |
| `OUTPUT_RETRIES`, `ENFORCE_VERDICTS` | the missing-verdict nudge (see design decisions) |
| `REPLAYS` | `2` runs the episode twice and reports whether the outputs are identical |

Tests:

```bash
cd code
python -m pytest
```

### Running with a real model

The constitution caps the LLM at about 20B parameters, so the intended targets are open
models served locally on the project GPU or hosted. A local server (vLLM, Ollama,
LM Studio, llama.cpp - all speak the OpenAI chat protocol):

```python
MODEL = "openai-chat:gpt-oss-20b"
MODEL_BASE_URL = "http://localhost:8000/v1"
MODEL_API_KEY_ENV = "OPENAI_API_KEY"       # whatever the server expects, or unset
```

Hosted or provider-native strings work the same way, with credentials in the provider's
usual environment variable:

```python
MODEL = "ollama:qwen3:14b"                 # OLLAMA_BASE_URL if not local
MODEL = "openrouter:openai/gpt-oss-20b"    # OPENROUTER_API_KEY
MODEL = "anthropic:claude-sonnet-4-5"      # ANTHROPIC_API_KEY - harness checks only, out of the 20B constraint
```

Free tiers work through the same `MODEL_BASE_URL` path. Both of these served the runs in
[End-to-end runs on free endpoints](#end-to-end-runs-on-free-endpoints); the second needs
a key:

```python
# OVHcloud AI Endpoints, anonymous: no key, 2 requests per minute per IP, and a rejected
# request counts against the window - so no SDK retries, and the harness waits for the
# retry-after the server announces
MODEL = "openai-chat:gpt-oss-20b"
MODEL_BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
MODEL_ANONYMOUS = True
REQUEST_INTERVAL_S = 35.0

# Z.AI (Zhipu), free GLM flash models: a key from the platform's console, kept in a file
# outside the repository (or in an environment variable named by MODEL_API_KEY_ENV).
# One request at a time: back-to-back requests get 429 without a retry-after, so the
# harness waits RATE_LIMIT_WAIT_S; a prompt beyond ~128k tokens is refused with HTTP 400
# "Prompt exceeds max length", hence the window below. PARALLEL_TOOL_CALLS = False is
# honoured (without it GLM batches 5-10 retrievals per turn)
MODEL = "openai-chat:glm-4.7-flash"
MODEL_BASE_URL = "https://api.z.ai/api/paas/v4"
MODEL_API_KEY_FILE = _REPO.parent / "zai_api_key.txt"
REQUEST_INTERVAL_S = 10.0
MAX_TOKENS = 8192
CONTEXT_TOKENS_LIMIT = 131_072
```

Provider-specific settings go into `EXTRA_MODEL_SETTINGS` (for example
`{"openai_reasoning_effort": "low"}`, or `{"extra_body": {"thinking": {"type":
"disabled"}}}` for GLM). `SEED` is honoured by OpenAI-style chat endpoints and vLLM and
ignored by providers that do not expose one; pydantic-ai drops `temperature` for the
newest Claude models, which do not accept it. `CONTEXT_TOKENS_LIMIT` should be set to the
served model's window so the episode ends with a recorded `budget` stop instead of a
provider error. With `REPLAYS = 2` the script says whether the two episodes agree, so the
determinism a given endpoint actually delivers is measured rather than assumed - hosted
APIs do not promise bit-exact greedy decoding.

## Outputs

One directory per run, `OUTPUT_DIR / RUN_NAME` (default
`<pool>[_<split>][_first<N>][_plainids]__<model slug>`, replays get a `__replay<k>` suffix):

**`anomaly_verdicts.jsonl`** (FR5) - one record per examined series, its standing verdict,
always in the pool's own ids:

```json
{"series_id": "R1__YAHOO__real_42_full", "verdict": "normal"}
{"series_id": "R1__SMD__machine-1-1_col_3_chunk2", "verdict": "anomalous", "explanation": "values jump from about 0.02 to 0.9 between time_index 1180 and 1215"}
```

`explanation` is present exactly when the verdict is `anomalous`. No score, no rank.

**`decision_trace.jsonl`** (FR6) - chronological, one typed record per line, `kind` says which:

| `kind` | One per | Carries |
|--------|---------|---------|
| `run_start` | run | run id, pool (name, repository-relative path, size, fingerprint), order seed, aliasing flag, model and settings, caps, prompt hash, registered tools, harness versions |
| `model_request` | LLM round trip | index, retry kind (`tool` / `output` / none), wall time, provider usage (input / output / cache tokens, details), context size in characters, tool calls requested, verdicts accepted / rejected |
| `tool_call` | tool invocation | tool, arguments as sent, resolved series id, ok or error text, points returned, payload size, wall time |
| `verdict` | parsed verdict | series id, verdict, whether the series was examined, accepted or the rejection reason, whether it superseded an earlier one, normalisation notes |
| `series` | series touched | `get_series` call count (and failures), tool time, first / last request, standing verdict, revisions, attributed token usage |
| `run_end` | run | termination reason, totals (requests, retries by kind, tool calls, tokens), overhead usage, coverage counts, verdict counts |

`termination.reason` is one of `completed` (the agent replied without a tool call),
`budget` (the harness projected that the next request would break `TOTAL_TOKENS_LIMIT` or
`CONTEXT_TOKENS_LIMIT` and stopped before fetching), `usage_limit` (a pydantic-ai cap
tripped after a response: request, tool-call or token count), `model_error` (the provider
failed), `protocol_error` (pydantic-ai gave up on the model's output: output retries
exhausted by the missing-verdict check, tool-argument retries exhausted, or a response
truncated before it produced anything) or `interrupted` (Ctrl-C or task cancellation).
Whatever the reason, the trace is closed by `run_end` and the verdicts file holds every
verdict accepted up to that point. `run_end.retries` counts retry prompts by kind: `output`
for the missing-verdict nudge and for pydantic-ai's own retry when a reply carried nothing
but reasoning, `tool` for invalid tool arguments and unknown tool names, `closing` for the
closing turn after a budget stop.

**`messages.json`** - the conversation as pydantic-ai kept it, tool payloads included: the
ground truth for anything about context. **`summary.json`** - the `run_end` record plus
identifiers. **`series_aliases.json`** - alias to pool id, absent only with `ALIAS_SERIES_IDS = False`.

`baseline_agent.schemas.validate_verdicts_file` / `validate_trace_file` validate the two
files strictly (types are checked, not coerced; the entry point runs both after every
episode). `baseline_agent.runner.compare_runs` compares two run directories field for
field, setting aside wall-clock fields (`started_at`, `finished_at`, `duration_ms`,
`tool_time_ms`), provider-generated ids, and the provider's cache accounting
(`cache_read_tokens`, `cache_write_tokens`, usage `details`), which a warm prompt cache
changes without changing what the model saw or said.

## Design decisions worth knowing

**Verdicts ride on the agent's text.** FR3 allows exactly two tools and FR5 wants a
verdict per examined series, produced as the episode goes. An output tool would be a third
tool exposed to the model and would end the run on first use, so the agent writes each
verdict as a one-line JSON object in its reply and the harness parses every response the
moment it arrives. The parser is lenient about placement (bare lines, code fences,
pretty-printed objects, arrays, prose around them) and strict about content, which is the
`VerdictRecord` schema; a line that tried to be a verdict object and failed is counted as
malformed, prose that merely mentions `series_id` is not. Episode rules on top: a verdict
counts only for a series whose values the agent retrieved in this episode, unknown ids are
discarded, the last verdict for a series wins, and restating a verdict changes nothing.
Every accepted, revised or discarded verdict is a `verdict` record in the trace, so
hallucinated ids and blind verdicts are measured instead of silently dropped.

**One protocol check, no coverage push.** If the agent ends the episode while a series it
retrieved has no verdict, the output validator answers with a `ModelRetry` naming the
series (bounded by `OUTPUT_RETRIES`, counted in `run_end.retries.output`). That is the FR4
protocol - a verdict for each examined series - and nothing more: the harness never asks
the agent to examine another series, never re-fetches, never compacts the history.
Stopping early is the baseline's own property and stays visible as
`series.examined` / `series.in_pool`. `ENFORCE_VERDICTS = False` switches even that check
off.

**One tool call at a time.** pydantic-ai runs tool calls of one response in parallel
threads by default, which makes the order of trace records - and the order in which
series count as examined - depend on thread timing whenever a model batches several
`get_series` calls into one turn. The tools are registered as sequential and the runner
pins sequential execution, so calls run in the order the model issued them;
`PARALLEL_TOOL_CALLS = False` additionally asks providers that support it for one call per
turn, which is idea B's turn-by-turn traversal (idea C, batched retrieval, was rejected by
the spec). A test batches five calls and checks three replays agree.

**The budget is enforced before the expensive request, not after it.** pydantic-ai checks
`total_tokens_limit` only once a response has been received and paid for, and in this
loop the request after a retrieval is the one that carries the new payload - so a cap that
trips there both overshoots the budget by a full context and discards the response that
held the verdict for the series just retrieved. After every response that asks for tools,
the runner projects the next request from the provider's own count of the current context,
the size of the requested payloads and a reply allowance, and ends the episode with reason
`budget` *before* fetching when the projection breaks `TOTAL_TOKENS_LIMIT` or
`CONTEXT_TOKENS_LIMIT`. The requested-but-unfetched ids are named in the termination
detail. When one turn asks for several series, they are admitted in issue order while the
projection still fits and the rest of the turn is answered with a textual `not retrieved`
error naming the series, so a model that batches retrievals (GLM-4.7-Flash does, whatever
`parallel_tool_calls` says) gets what fits instead of ending the episode on its first
over-sized turn; only a turn whose first retrieval does not fit is a budget stop. If the
model still owes verdicts for series it did retrieve at that point (some
models fetch several series before judging any), it gets one closing turn: the pending
call is answered with a not-executed error and the model is asked for those verdicts -
the same examined-implies-judged rule the output validator applies when the model stops by
itself, never a request to examine more. The closing turn is skipped when it would not
fit the cap either, and everything about it is in the trace (`retry_kind = "closing"`, the
unexecuted call as a failed `tool_call`, `run_end.retries.closing`). pydantic-ai's own
caps stay on as the backstop, and if one of them does trip after a response, the response
is still booked exactly: the model is wrapped so the harness keeps what pydantic-ai drops.

**Label-free by construction, and no provenance in the ids.** `Pool.from_parquet` reads
`series_id`, `time_index`, `value` and nothing else; the metadata file is opened only to
select a split, and only its `series_id` / `split` columns; the prompt names no base rate,
pool size or series length. The spec-01 id itself would leak: on R1 its provenance
predicts the label - YAHOO samples are 95% anomalous, NEK 100%, unsplit `_full` samples
96% against 23% for chunks (R2 is milder: 20-36% by source) - so an agent could guess
from the id without reading a value, and in the first runs with plain ids gpt-oss-20b
visibly chose series by name ("maybe pick WSD series"). The model therefore sees opaque
`s0001`-style names, numbered in served order, in both tools and in the retry message;
every output file and trace field carries the pool's own ids, the mapping is written next
to the outputs (`series_aliases.json`), and the `get_pool_meta_data` payload shrinks about
five-fold as a side effect. `ALIAS_SERIES_IDS = False` serves the pool's ids as they are,
for experiments on the leak itself, and marks the run name with `plainids`. Whichever ids
the baseline sees, the tool-augmented agent of M2/M3 must be served the same.

**Served order is a seed, not the file order.** Spec-01 files are grouped by source, so
file order walks one recording at a time: the first 25 R1 series are consecutive Exathlon
chunks with a 4% anomaly rate, and the positive rate climbs from 9% in the first decile of
the file to 45% in the last. With a budget that covers a dozen series, an agent working
down the list would judge an unrepresentative slice. `ORDER_SEED` (42 in the entry point)
serves a seeded permutation, recorded in `run_start` and in the run id; `SERIES_LIMIT`
applies to the served order, so a "first N" is a seeded sample. `None` restores file order.

**Tool results are text with a pinned wire format.** Both tools return the JSON string
the model will see (compact, no whitespace, insertion-ordered keys, integers kept as
integers) rather than a dict for pydantic-ai to serialise. That makes the payload size a
fact (`payload_chars` in the trace), keeps replays byte-identical across providers, and is
where FR1's "textual error message" lives: an unknown id or a broken series (NaN,
non-contiguous index) yields `Error: ...` for the model to read, never an exception.

**Every ending is a recorded ending.** The runner steps through pydantic-ai's graph
(`agent.iter`) so that each model request is timed and its usage booked as it happens,
and verdicts and trace lines are flushed to disk immediately. A budget stop, a usage cap,
a provider error, exhausted retries or a cancellation all end in a `run_end` record with
the reason, and every file is written. `history_chars` counts the system prompt once per
request, as the provider receives it, not once per request kept in the history.

**Token attribution.** Providers report usage per request, and one request can consume
the payloads of several series. Each request is charged to the series whose `get_series`
payloads it consumed (the successful calls executed right before it), split equally;
requests that follow no retrieval - the prompt-only first turn, the metadata turn, retries,
a closing message sent without a retrieval before it - are booked as `overhead_usage`.
Shares are fractional so that `series.attributed_usage` plus `run_end.overhead_usage`
equal `run_end.usage` exactly; a test checks the sum.

**Determinism (FR7).** Fixed prompt, seeded served order, greedy decoding with a seed,
sequential tool execution, no timestamps in anything the model sees, deterministic tool
payloads. The run id is a hash of the pool contents and the configuration, so two replays
of the same setup share it. What legitimately differs is wall-clock timing,
provider-generated ids and cache accounting, and `compare_runs` sets exactly those aside.
`messages.json` is compared too, so the check covers the whole conversation.

**No history compaction, no batching, no rounding.** The natural fixes for the context
arithmetic below - dropping old payloads, fetching several series per call, sending fewer
digits - would each change what the baseline is; spec 02 rejects batching (idea C) and RQ3
is about tool outputs that avoid this cost in the first place. The harness keeps every
payload in context verbatim. `VALUE_PRECISION` exists for experiments on the last point and
is off by default.

**The scripted stand-in is a model, not a method.** `MODEL = "scripted"` plugs a
deterministic policy into pydantic-ai's `FunctionModel`: discover the pool, retrieve
series in served order, judge each with a robust-z / frozen-run rule, stop. It has switches
to misbehave on purpose (blind verdicts, unknown ids, malformed lines, never stopping,
skipping a verdict, quitting at once) so the tests can cover the harness's answers. Its
token counts are estimates at 4 characters per token - optimistic for numeric JSON, which
tokenises at under 2 - and it is labelled `scripted` everywhere.

## End-to-end runs on free endpoints

The E2E checks of spec §3 were run on `gpt-oss-20b` served by OVHcloud's anonymous AI
Endpoints tier (no key; 2 requests per minute per IP with rejected requests counted, so
about one request per minute in practice; `parallel_tool_calls` unsupported;
`reasoning_effort` supported). Configuration: opaque ids, `ORDER_SEED = 42`, temperature 0,
seed 42, `reasoning_effort = "low"`, `MAX_TOKENS = 32768`, `CONTEXT_TOKENS_LIMIT = 131072`,
`REQUEST_INTERVAL_S = 62`, no SDK retries, harness retries on 429 with the announced
`retry-after`. Verdicts, traces, summaries and alias maps are kept under
[`results/`](results/) (`messages.json` with the raw payloads is not);
`src/summarize_run.py` prints the numbers below from a run directory.

The runs with opaque ids at a glance (gpt-oss-20b on OVHcloud, GLM-4.7-Flash on Z.AI's
free tier; the details follow):

| Run | Model | Requests | Series retrieved / judged | Termination | Tokens | Wall | Against the labels |
|-----|-------|----------|---------------------------|-------------|--------|------|--------------------|
| full R1 | gpt-oss-20b | 3 | 1 / 1 | `completed` | 59.3k | 141 s | 1 false positive |
| full R2 | gpt-oss-20b | 3 | 1 / 1 | `completed` | 35.9k | 128 s | 1 miss |
| dev-5, episode 1 | gpt-oss-20b | 6 | 3 / 3 | `completed` | 150.5k | 322 s | 3 correct `normal` |
| dev-5, episode 2 | gpt-oss-20b | 4 | 1 / 1 | `completed` | 28.0k | 245 s | 1 false positive |
| full R1 | GLM-4.7-Flash | 3 | 4 / 4 | `budget` | 134.8k | 118 s | 4 correct `normal` |
| full R2 | GLM-4.7-Flash | 4 | 10 / 10 | `budget` | 246.7k | 634 s | 1 hit, 4 misses, 5 correct `normal` |
| dev-5, episode 1 | GLM-4.7-Flash | 5 | 4 / 3 | `model_error` (429) | 111.4k | 1,008 s | 2 false positives, 1 correct `normal` |
| dev-5, episode 2 | GLM-4.7-Flash | 5 | 4 / 3 | `model_error` (500) | 115.2k | 1,432 s | 1 false positive, 2 correct `normal` |

**Full R1 run**: three requests. The id list brings the second request to 12.6k tokens
(53.6k with the pool's own ids); the model decides to "pick few", takes the first served
id - `s0001`, which is `R1__WSD__WSD_90_chunk1`, 4,320 points, 33k tokens of payload -
and after a 17-second reply judges it `anomalous` ("hard to parse, but we can spot
anomalies"): a level shift "after index 2000 from ~1,000 to >5,000", in a series the
labels call normal. Then it stops. Termination `completed`, 1 of 2,900 series examined
and judged, 59.3k tokens, 141 s.

**Full R2 run**: the same three requests. `s0001` is `R2__SVDB__874_col_1_chunk13`
(1,520 points, label 1), judged `normal` ("clear periodic peaks ... looks normal") - a
false negative. Termination `completed`, 1 of 2,900, 35.9k tokens, 128 s.

**Dev-subset pair** (R1 `dev` split, served order seed 42, first 5 series, two episodes).
The first episode retrieved three series in a row without judging any ("Huge. Hard.
Probably normal. I'll skip."), replied with one verdict, and delivered the other two on
the missing-verdict nudge: three `normal` verdicts on three normal series, 6 requests,
150.5k tokens, 322 s. The second episode diverged at the third request: "DONE 0" right
after the first retrieval, one nudge, then 4.9k tokens of reasoning ending in `anomalous`
with spikes cited at index 1200-1201, for a series the labels call normal: 4 requests,
28.0k tokens, 245 s.

**Determinism (FR7).** No replay pair agreed, with opaque ids or with the pool's own: at
identical prompts, temperature 0 and `seed = 42` the model's reasoning text differs from
the second or third request on, and the episodes diverge (one fetches another series, the
other replies with text). The harness itself replays identically (scripted model,
byte-identical verdicts and traces), so this is the endpoint's property - batched
inference on a free tier - and `compare_runs` reports it instead of hiding it. FR7
therefore holds for the harness and has to be re-checked per endpoint; a locally served
model with a fixed batch size is the way to get it end to end.

**What the harness does.** Across all episodes every ending was a recorded one with closed,
valid files: `completed`, `budget` (the projection refused a fourth, ~49k-token payload
that would not fit the window), `protocol_error` (reasoning cut by `max_tokens`) and
`model_error` (timeouts). The missing-verdict check fired in four episodes and recovered
verdicts in three. Stray tool arguments (`get_pool_meta_data({"": ""})`, sent by gpt-oss
on every first turn) cost a retry until the tools were made to ignore them.

### A second endpoint: GLM-4.7-Flash on Z.AI's free tier

The same episodes were run on `glm-4.7-flash` through Z.AI's free tier (a key from the
platform's console; one request at a time, 429 without `retry-after` on back-to-back
requests; thinking left at the model's default; `MAX_TOKENS = 8192`,
`CONTEXT_TOKENS_LIMIT = 131072` after a probe had a ~150k-token prompt refused with
HTTP 400 "Prompt exceeds max length" while 119k went through; `REQUEST_INTERVAL_S = 10`;
`PARALLEL_TOOL_CALLS = False`). The tier is accepted but unreliable: requests sat in its
queue for one to four minutes, two episodes ended in a 429 "service temporarily
overloaded" or a 500 "network error" that outlasted the harness's eight retries, one
1k-token request hit the 900 s timeout, and one full-R1 attempt never got a first
response in fourteen minutes of retries.

**What the model does with raw series.** The opposite of gpt-oss: GLM walks the served
list in order, reasons for 150-5,000 tokens over each payload ("Let me analyze s0001. The
values range from around 21 to 28 ..."), writes the verdict and the next retrieval in one
reply, as the protocol asks, and keeps going until the window is full. It also ignores
`parallel_tool_calls`: on the full pools it asks for five series per turn ("given the
large number of series, I need to be efficient"), which is where the prefix rule of the
budget projection matters - a probe run made before that rule ended with zero series
examined, because its first five-series turn (~121k tokens of payload) did not fit as a
whole.

**Dev-subset pair** (same five series, two episodes): both retrieved four series one at a
time and judged three before the tier failed the sixth request (429 overloaded in one,
500 in the other). The verdicts flipped between the episodes - `normal` / `anomalous` /
`anomalous` in the first, `anomalous` / `normal` / `normal` in the second, on the same
three normal series, the explanations citing "sustained level shifts" either way: 111k
and 115k tokens, 1,008 s and 1,432 s.

**Full R2 run**: four requests, five retrievals per turn. Ten series judged - one
`anomalous`, a true positive (`R2__MITDB__215_col_0_chunk154`); four positives and five
negatives called `normal` - then the projection refused the third batch (52k tokens of
payload on top of a 123k-token context) and the episode ended with reason `budget`, every
retrieved series judged. The fourth reply carried 38k tokens of reasoning over five ECG
chunks: `MAX_TOKENS` caps the visible reply, not the thinking. 246.7k tokens, 634 s.

**Full R1 run**: three requests. The model asks for five series at once; the projection
admits the first four (105k tokens of payload on top of a 14k-token context) and refuses
the fifth, the model reads the refusal ("I see that s0005 would not fit in the budget"),
judges the four it got - `normal` on four normal series - and asks for four more, which
the projection refuses as a whole (170k tokens against a 131k window): reason `budget`,
4 of 2,900 series examined and judged, 134.8k tokens, 118 s.

**Determinism (FR7)**: no pair agreed here either; the two dev-subset episodes differ
from the very first response (the reasoning text is not the same), and their verdicts
disagree on every series.

### Before the switch to opaque ids

The first runs served the pool's own ids (kept under `results/` with a `plainids` marker)
and showed why that is a leak: in the reasoning the model picks by name ("Huge. Hard.
Maybe pick WSD series.", then "let's pick a manageable one" - `R1__WSD__WSD_90_chunk0`),
and the id list alone made every request 53.6k tokens heavier on R1 (38.9k on R2). They
also set the decoding knobs: with the default reasoning effort the model spends its whole
reply reasoning over the 1,500 numbers it retrieved - the first attempt hit
`MAX_TOKENS = 8192` inside the reasoning and produced no reply at all, the second produced
one verdict after 3,250 reasoning tokens, and with `MAX_TOKENS = 32768` two attempts in a
row exceeded a 300 s timeout on the very first judging request - hence
`reasoning_effort = "low"`, where replies come in 40-70 output tokens.

| Run (plain ids) | Effort | Requests | Series retrieved / judged | Termination | Tokens (in / out) | Wall |
|-----|--------|----------|---------------------------|-------------|-------------------|------|
| dev pair 1, episode 1 | medium, `MAX_TOKENS` 8192 | 4 | 1 / 0 | `protocol_error` (reasoning cut) | 13.6k / 8.3k | 234 s |
| dev pair 1, episode 2 | medium, `MAX_TOKENS` 8192 | 4 | 1 / 1 (`anomalous`, label 0) | `completed` | 13.6k / 3.4k | 254 s |
| dev pair 2, both episodes | medium, `MAX_TOKENS` 32768, timeout 300 s | 2-3 | 1-2 / 0 | `model_error` (timeout) | - | ~500 s |
| dev pair 3, episode 1 | low | 5 | 3 / 0 | `budget` (4th payload would not fit) | 93.5k / 0.3k | 303 s |
| dev pair 3, episode 2 | low | 3 | 1 / 1 (`normal`, label 0) | `completed` | 12.6k / 0.1k | 173 s |
| dev pair 4, episode 1 | low, closing turn on | 5 | 3 / 1 (`normal`, label 0) | `model_error` (57k-token nudge timed out) | 93.4k / 0.2k | 1,343 s |
| dev pair 4, episode 2 | low, closing turn on | 4 | 1 / 1 (`normal`, label 0) | `completed` (after the nudge) | 23.3k / 0.2k | 188 s |
| full R1 | low | 3 | 1 / 1 (`anomalous`, label 0) | `completed` | 141.2k / 0.5k | 206 s |
| full R2 | low | 4 | 1 / 1 (`anomalous`, label 1, citing the QRS complex) | `completed` (after pydantic-ai's own retry) | 137.0k / 11.3k | 724 s |

That is the raw-retrieval baseline on real pools: within the constitution's daily budget it
covers one series per episode with gpt-oss-20b, which judges at a glance, and three to ten
with GLM-4.7-Flash, which reasons over every payload until the window is full; its
verdicts on the same series flip between episodes, and the trace says exactly where the
tokens went - with opaque ids the id list, sent with every request, is still 40% of the
gpt-oss R1 episode and two thirds of its R2 one, the rest is one payload.

## What to expect from a full run

`get_series` returns the whole series, `get_pool_meta_data` lists every id, and in a single
episode every request re-sends the whole history. Measured on the four pools with the
o200k tokenizer (the GPT-4o / gpt-oss family; `src/measure_payloads.py`):

| Pool | Series | Points / series (median / max) | `get_series` payload, chars (median / max) | Chars / token | Tokens / series (median) | `get_pool_meta_data` payload (aliased ids) | Whole pool once | Series per episode (1M budget) | Series per request (128k window) |
|------|--------|--------------------------------|--------------------------------------------|---------------|--------------------------|------------------------------|-----------------|--------------------------------|----------------------------------|
| R1 | 2,900 | 1,500 / 12,960 | 24.5k / 231.6k | 1.73 | ~14.2k | 23.2k chars ~ 11.6k tokens | ~55M tokens | ~10 | ~8 |
| R2 | 2,900 | 1,536 / 1,696 | 15.7k / 18.4k | 1.60 | ~9.8k | 23.2k chars ~ 11.6k tokens | ~29M tokens | ~12 | ~11 |
| S1 | 2,000 | 2,519 / 4,000 | 59.1k / 97.4k | 2.20 | ~26.9k | 16.0k chars ~ 8.0k tokens | ~54M tokens | ~7 | ~4 |
| S2 | 2,000 | 2,526 / 3,999 | 59.3k / 97.6k | 2.21 | ~26.8k | 16.0k chars ~ 8.0k tokens | ~54M tokens | ~7 | ~4 |

Numeric JSON tokenises at under two characters per token, the synthetic pools carry
full-precision `float64` values (about 23 characters per point, twice the real pools), and
the id list of a 2,900-series pool is ~12k tokens that every request repeats (with the
pool's own ids it would be 40-50k: R1's is 118.6k characters, ~52.6k tokens). After k
series of R1 a request carries roughly 11.6k + 14.2k × k tokens and the episode has
consumed roughly 11.6k × (k+1) + 14.2k × k(k+1)/2:

- the constitution's budget of about 1M tokens is spent after ~10 R1 series (~12 on R2,
  ~7 on S1/S2); with the pool's own ids it would be ~8 (~10, ~6-7);
- a 128k-token window holds the id list plus ~8 R1 series, and the largest R1 series
  alone (12,960 points, ~134k tokens) does not fit in it at all.

The dry run with the stand-in (4 characters per token, so optimistic) on R1 with
`ORDER_SEED = 42` ended with reason `budget` after 16 requests: 14 series retrieved, all 14
judged, 967,383 estimated tokens spent, the 15th retrieval refused because the projection
reached 1.14M. With the pool's own ids it managed 11 series in 13 requests for 906,634.
Both replay identically.

So on this baseline "a full run over one real dataset" is a run that terminates cleanly
with every examined series judged and the caps respected, not one that reaches every
series: coverage of a 2,900-series pool is two orders of magnitude out of reach in a single
episode, which is precisely the property RQ1 and RQ3 are set up to quantify. The trace
records the coverage actually achieved and what it cost. For cheaper experiments,
`SPLIT = "dev"` and `SERIES_LIMIT` restrict the pool while keeping the episode's
single-invocation shape.
