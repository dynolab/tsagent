"""Summarise one or more baseline-agent runs: coverage, cost, termination, and - when the
pool's spec-01 metadata is at hand - how the verdicts on the examined series compare with
the series-level labels.

The label comparison is a sanity read-out for the write-up, not the RQ1 evaluation: the
examined subset is small and chosen by the agent, so nothing here is a benchmark number.

Every knob lives in the CONFIGURATION block; no command-line arguments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "code"))

import pyarrow.parquet as pq  # noqa: E402

from src.baseline_agent.runner import compare_runs  # noqa: E402
from src.baseline_agent.schemas import ModelRequestRecord, RunEndRecord, RunStartRecord, SeriesRecord, validate_trace_file, validate_verdicts_file  # noqa: E402

# ======================================================================================
# CONFIGURATION
# ======================================================================================

RUNS_DIR = _REPO / "data" / "02-raw-retrieval-baseline-agent-infrastructure" / "runs"
RUN_NAMES: list[str] = []          # run directories under RUNS_DIR; [] = every directory there
_DATA_DIRS = [_REPO / "data" / "01-anomaly-detection-agent-spec", _REPO / "data" / "anomaly_detection"]
COMPARE_REPLAYS = True             # compare <name> with <name>__replay<k> when both exist

# ======================================================================================


def metadata_labels(pool_name: str) -> dict[str, int] | None:
    for base in _DATA_DIRS:
        for sub in ("real", "synthetic", "full"):
            path = base / sub / f"{pool_name}_metadata.parquet"
            if path.is_file():
                table = pq.read_table(path, columns=["series_id", "y_i"])
                return dict(zip(table.column("series_id").to_pylist(), table.column("y_i").to_pylist()))
    return None


def summarise(run_dir: Path) -> dict:
    records = validate_trace_file(run_dir / "decision_trace.jsonl")
    verdicts = validate_verdicts_file(run_dir / "anomaly_verdicts.jsonl")
    start: RunStartRecord = records[0]
    end: RunEndRecord = records[-1]
    requests = [r for r in records if isinstance(r, ModelRequestRecord)]
    series = [r for r in records if isinstance(r, SeriesRecord)]
    out = {
        "run": run_dir.name,
        "pool": start.pool.name,
        "model": start.model.name,
        "order_seed": start.order_seed,
        "aliased": start.series_id_aliasing,
        "termination": end.termination.reason,
        "requests": end.requests,
        "retries": end.retries.model_dump(),
        "get_series_calls": end.tool_calls.get_series,
        "failed_tool_calls": end.tool_calls.failed,
        "examined": end.series.examined,
        "verdicted": end.series.verdicted,
        "examined_without_verdict": end.series.examined_without_verdict,
        "anomalous": end.series.anomalous,
        "verdicts_rejected": end.verdicts.rejected,
        "verdicts_revised": end.verdicts.revised,
        "input_tokens": end.usage.input_tokens,
        "output_tokens": end.usage.output_tokens,
        "total_tokens": end.usage.total_tokens,
        "max_context_tokens": max((r.usage.input_tokens for r in requests), default=0),
        "max_history_chars": max((r.history_chars for r in requests), default=0),
        "wall_s": round(end.duration_ms / 1000.0, 1),
        "mean_request_s": round(sum(r.duration_ms for r in requests) / 1000.0 / max(1, len(requests)), 1),
        "tokens_per_examined_series": round(end.usage.total_tokens / end.series.examined, 1) if end.series.examined else None,
    }
    labels = metadata_labels(start.pool.name)
    if labels is not None:
        tp = fp = tn = fn = 0
        for v in verdicts:
            y = labels.get(v.series_id)
            if y is None:
                continue
            if v.verdict == "anomalous":
                tp += y == 1
                fp += y == 0
            else:
                tn += y == 0
                fn += y == 1
        out["labels"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "positives_examined": tp + fn}
        explained = [v for v in verdicts if v.verdict == "anomalous"]
        out["explanation_chars_mean"] = round(sum(len(v.explanation or "") for v in explained) / len(explained), 1) if explained else None
    out["series"] = [
        {"series_id": s.series_id, "verdict": s.verdict, "label": (labels or {}).get(s.series_id), "calls": s.get_series_calls, "tokens": round(s.attributed_usage.total_tokens)}
        for s in series
    ]
    return out


def main() -> None:
    names = RUN_NAMES or sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())
    seen: set[str] = set()
    for name in names:
        run_dir = RUNS_DIR / name
        if not (run_dir / "decision_trace.jsonl").is_file():
            continue
        summary = summarise(run_dir)
        seen.add(name)
        print("=" * 100)
        for key, value in summary.items():
            if key == "series":
                continue
            print(f"{key:>26}: {value}")
        for s in summary["series"]:
            print(f"    {s['series_id']:<52} verdict={str(s['verdict']):<9} label={s['label']} calls={s['calls']} tokens={s['tokens']}")
        if COMPARE_REPLAYS and "__replay" not in name:
            for other in names:
                if other.startswith(name + "__replay") and (RUNS_DIR / other / "decision_trace.jsonl").is_file():
                    diff = compare_runs(run_dir, RUNS_DIR / other)
                    verdict = "identical" if diff.identical else "DIFFERENT: " + "; ".join(diff.differences)
                    print(f"    replay {other}: {verdict}")
    if not seen:
        print(f"no runs found under {RUNS_DIR}")


if __name__ == "__main__":
    main()
