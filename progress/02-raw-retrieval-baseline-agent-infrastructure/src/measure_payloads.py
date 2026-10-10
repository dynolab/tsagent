"""Measure what the two tools put into the model's context, pool by pool.

Prints the table behind the "What to expect from a full run" section of the task README:
per pool, the size of a ``get_series`` payload (characters and tokens), the size of the
``get_pool_meta_data`` payload that every request re-sends, and the number of series a
single episode can afford under a token budget and a context window. Tokens are counted
with tiktoken's ``o200k_base`` (the GPT-4o / gpt-oss family) when tiktoken is installed;
otherwise only characters are reported.

Every tunable knob lives in the CONFIGURATION block below; no command-line arguments.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "code"))

from src.baseline_agent.ids import IdAliases  # noqa: E402
from src.baseline_agent.pool import Pool  # noqa: E402
from src.baseline_agent.tools import get_pool_meta_data, get_series  # noqa: E402

# ======================================================================================
# CONFIGURATION
# ======================================================================================

_DATA_DIRS = [_REPO / "data" / "01-anomaly-detection-agent-spec", _REPO / "data" / "anomaly_detection"]
POOLS = ["real/R1", "real/R2", "synthetic/S1", "synthetic/S2"]
TOKENIZER = "o200k_base"       # tiktoken encoding; ignored when tiktoken is not installed
SAMPLE_SERIES = 40             # series per pool tokenised to estimate characters per token
ALIAS_SERIES_IDS = True        # measure the id list the model sees by default (opaque aliases);
                               # the pool's own ids are reported alongside either way
TOKEN_BUDGET = 1_000_000       # constitution budget for one episode
CONTEXT_WINDOW = 128_000       # a common window size, for the "how many series fit" column
OUTPUT_ALLOWANCE = 1_024       # reply size assumed per request

# ======================================================================================


def data_dir() -> Path:
    for base in _DATA_DIRS:
        if base.is_dir():
            return base
    raise FileNotFoundError(f"no pool directory found; tried {[str(p) for p in _DATA_DIRS]}")


def tokenizer():
    try:
        import tiktoken
    except ImportError:
        return None
    return tiktoken.get_encoding(TOKENIZER)


def affordable_series(meta_tokens: float, series_tokens: float, budget: float) -> int:
    """Largest k with cumulative cost sum_{j=1..k+1} (meta + j*series + allowance) <= budget.

    Request j+1 carries the metadata payload, j series payloads and the reply so far; the
    closed form is meta*(k+1) + series*k(k+1)/2 + allowance*(k+1).
    """
    k = 0
    while True:
        cost = (meta_tokens + OUTPUT_ALLOWANCE) * (k + 2) + series_tokens * (k + 1) * (k + 2) / 2
        if cost > budget:
            return k
        k += 1


def fits_window(meta_tokens: float, series_tokens: float, window: float) -> int:
    """Largest k such that a request carrying the metadata payload and k series fits."""
    return max(0, int((window - meta_tokens - OUTPUT_ALLOWANCE) // series_tokens))


def main() -> None:
    enc = tokenizer()
    base = data_dir()
    rows = []
    for rel in POOLS:
        pool = Pool.from_parquet(base / f"{rel}.parquet")
        points = [pool.length_of(s) for s in pool.series_ids]
        chars = [len(get_series(pool, s)) for s in pool.series_ids]
        served = IdAliases(pool.series_ids).served if ALIAS_SERIES_IDS else pool.series_ids
        meta = get_pool_meta_data(pool, series_ids=served)
        meta_plain = get_pool_meta_data(pool)
        idx = [int(i * (len(pool) - 1) / max(1, SAMPLE_SERIES - 1)) for i in range(SAMPLE_SERIES)]
        if enc is not None:
            ratio = statistics.mean(len(get_series(pool, pool.series_ids[i])) / len(enc.encode(get_series(pool, pool.series_ids[i]))) for i in idx)
            meta_tokens = len(enc.encode(meta))
            meta_plain_tokens = len(enc.encode(meta_plain))
        else:
            ratio = float("nan")
            meta_tokens = float("nan")
            meta_plain_tokens = float("nan")
        median_chars = statistics.median(chars)
        series_tokens = median_chars / ratio if enc is not None else float("nan")
        rows.append(
            {
                "pool": pool.name,
                "series": len(pool),
                "points_median": int(statistics.median(points)),
                "points_max": max(points),
                "payload_chars_median": int(median_chars),
                "payload_chars_max": max(chars),
                "chars_per_token": ratio,
                "tokens_per_series": series_tokens,
                "meta_chars": len(meta),
                "meta_tokens": meta_tokens,
                "meta_plain_chars": len(meta_plain),
                "meta_plain_tokens": meta_plain_tokens,
                "pool_once_tokens": sum(chars) / ratio if enc is not None else float("nan"),
                "series_in_budget": affordable_series(meta_tokens, series_tokens, TOKEN_BUDGET) if enc is not None else None,
                "series_in_window": fits_window(meta_tokens, series_tokens, CONTEXT_WINDOW) if enc is not None else None,
            }
        )

    print(f"tokenizer: {TOKENIZER if enc is not None else 'none (tiktoken not installed)'}; budget {TOKEN_BUDGET:,}; window {CONTEXT_WINDOW:,}; reply allowance {OUTPUT_ALLOWANCE:,}; ids: {'opaque aliases' if ALIAS_SERIES_IDS else 'the pool''s own'}")
    header = "| Pool | Series | Points / series (median / max) | get_series payload, chars (median / max) | Chars / token | Tokens / series (median) | get_pool_meta_data payload | Whole pool once | Series per episode (budget) | Series per request (window) |"
    print(header)
    print("|" + "|".join("---" for _ in header.split("|")[1:-1]) + "|")
    for r in rows:
        print(
            f"| {r['pool']} | {r['series']:,} | {r['points_median']:,} / {r['points_max']:,} | "
            f"{r['payload_chars_median'] / 1000:.1f}k / {r['payload_chars_max'] / 1000:.1f}k | {r['chars_per_token']:.2f} | "
            f"~{r['tokens_per_series'] / 1000:.1f}k | {r['meta_chars'] / 1000:.1f}k chars ~ {r['meta_tokens'] / 1000:.1f}k tokens | "
            f"~{r['pool_once_tokens'] / 1e6:.0f}M tokens | ~{r['series_in_budget']} | ~{r['series_in_window']} |"
        )
    print("get_pool_meta_data with the pool's own ids: " + "; ".join(f"{r['pool']} {r['meta_plain_chars'] / 1000:.1f}k chars ~ {r['meta_plain_tokens'] / 1000:.1f}k tokens" for r in rows))


if __name__ == "__main__":
    main()
