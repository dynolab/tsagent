"""Series ids as the model sees them.

Spec-01 ids carry their provenance - ``R1__YAHOO__real_42_full`` - and on R1 that
provenance predicts the label: YAHOO samples are 95% anomalous, NEK 100%, unsplit
``_full`` samples 96% against 23% for chunks. A raw-retrieval agent could therefore guess
from the id without reading a value. So by default both tools speak opaque names
(``s0001`` ... in served order) and the agent writes verdicts against them; every output
file and every trace field still carries the pool's ids, and the mapping is written next
to them. ``PlainIds`` serves the pool's ids as they are, for experiments on the leak
itself; the flag is recorded in ``run_start`` either way (see the task README).

Both views resolve what the model wrote back to a pool id, tolerating surrounding
whitespace, and return ``None`` for anything else.
"""

from __future__ import annotations

from typing import Sequence


class PlainIds:
    """The pool's own ids."""

    aliased = False

    def __init__(self, series_ids: Sequence[str]):
        self._ids = list(series_ids)
        self._known = set(self._ids)

    @property
    def served(self) -> list[str]:
        """What ``get_pool_meta_data`` lists, in order."""
        return list(self._ids)

    def display(self, real_id: str) -> str:
        return real_id

    def resolve(self, model_id: str) -> str | None:
        if model_id in self._known:
            return model_id
        trimmed = model_id.strip()
        return trimmed if trimmed in self._known else None

    def mapping(self) -> dict[str, str] | None:
        return None


class IdAliases:
    """Opaque ``s0001``-style names, numbered in served order."""

    aliased = True

    def __init__(self, series_ids: Sequence[str]):
        self._ids = list(series_ids)
        width = max(4, len(str(len(self._ids))))
        self._to_alias = {sid: f"s{i:0{width}d}" for i, sid in enumerate(self._ids, start=1)}
        self._to_real = {alias: sid for sid, alias in self._to_alias.items()}

    @property
    def served(self) -> list[str]:
        return [self._to_alias[sid] for sid in self._ids]

    def display(self, real_id: str) -> str:
        return self._to_alias.get(real_id, real_id)

    def resolve(self, model_id: str) -> str | None:
        real = self._to_real.get(model_id)
        if real is None:
            real = self._to_real.get(model_id.strip())
        return real

    def mapping(self) -> dict[str, str]:
        """alias -> pool id, in served order."""
        return {self._to_alias[sid]: sid for sid in self._ids}
