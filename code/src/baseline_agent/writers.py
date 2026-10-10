"""Append-only JSONL writers for the episode outputs.

Both files are written as the episode unfolds, one flushed line per event, so an
interrupted run still leaves a readable prefix behind.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from .schemas import VerdictRecord


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj


def dumps_line(obj: Any) -> str:
    """One JSON line: UTF-8 verbatim, keys in insertion order, no trailing whitespace."""
    return json.dumps(to_jsonable(obj), ensure_ascii=False, separators=(", ", ": "))


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class JsonlWriter:
    """Appends one JSON object per line, flushing after every write."""

    def __init__(self, path: str | Path, *, truncate: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w" if truncate else "a", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, obj: Any) -> None:
        self._fh.write(dumps_line(obj) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VerdictWriter:
    """Keeps ``anomaly_verdicts.jsonl`` equal to the standing verdict per series.

    New verdicts are appended; a revised verdict (the agent changed its mind about a series
    it had already judged) rewrites the file in place, atomically, so the file never holds
    two records for one series.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._final: OrderedDict[str, VerdictRecord] = OrderedDict()
        self._rewrite()

    def put(self, record: VerdictRecord) -> bool:
        """Store ``record``; returns True when it replaced an earlier verdict."""
        supersedes = record.series_id in self._final
        self._final[record.series_id] = record
        if supersedes:
            self._rewrite()
        else:
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(self._line(record) + "\n")
        return supersedes

    @staticmethod
    def _line(record: VerdictRecord) -> str:
        # ``explanation`` is present exactly when the verdict is anomalous (FR5).
        return dumps_line(record.model_dump(mode="json", exclude_none=True))

    def put_all(self, records: Iterable[VerdictRecord]) -> int:
        return sum(self.put(r) for r in records)

    def records(self) -> list[VerdictRecord]:
        return list(self._final.values())

    def __len__(self) -> int:
        return len(self._final)

    def _rewrite(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for rec in self._final.values():
                fh.write(self._line(rec) + "\n")
        os.replace(tmp, self.path)
