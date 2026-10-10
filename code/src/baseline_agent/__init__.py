"""Raw-retrieval-only baseline agent (spec 02).

One pydantic-ai agent, two tools (``get_series``, ``get_pool_meta_data``), one episode
over a whole pool, two output files (``anomaly_verdicts.jsonl``, ``decision_trace.jsonl``).

Entry points:

- :func:`baseline_agent.runner.run_episode` - run one episode and write its outputs
- :func:`baseline_agent.runner.compare_runs` - field-for-field replay check (FR7)
- ``src/run_baseline_agent.py`` - configuration-driven script wiring it all together
"""

from .agent import TOOL_NAMES, build_agent, registered_tool_names
from .pool import Pool, SeriesData, read_split_ids
from .schemas import VerdictRecord, validate_trace_file, validate_verdicts_file
from .tools import get_pool_meta_data, get_series

__all__ = [
    "TOOL_NAMES",
    "Pool",
    "SeriesData",
    "VerdictRecord",
    "build_agent",
    "get_pool_meta_data",
    "get_series",
    "read_split_ids",
    "registered_tool_names",
    "validate_trace_file",
    "validate_verdicts_file",
]
