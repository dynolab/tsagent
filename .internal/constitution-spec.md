## tsagent constitution spec

### 1. Executive summary

#### 1.1 Research topic

Evidence-Grounded LLM Agent-Based Orchestration of Time Series Analysis Tools (Оркестрация алгоритмов анализа временных рядов на основе LLM-агента)

#### 1.2 Abstract

Build an offline LLM-based agent that helps users triage large pools of univariate time series by planning and orchestrating classical time-series analysis tools (anomaly detection, change point detection, decomposition, forecasting, similarity search) and producing concise natural-language reports. The agent has no direct access to raw time series — it can only retrieve series data and analysis results through deterministic tools, each returning structured outputs. These outputs include evidence links: precise, structured references to specific data points (exact timestamps, contiguous segments, summary statistics) that ground every claim in the generated narrative. Industrial monitoring is a primary inspiration, but the approach should transfer to IoT, finance, web metrics, and medicine.

#### 1.3 Motivation

Large-scale time series systems generate far more data than manual investigation can handle, yet one-size-fits-all detectors produce excessive alerts. LLMs excel at planning and explanation but are prone to hallucination when applied to raw data. An evidence-grounded tool interface that constrains the LLM to cite only structured tool outputs — including precise evidence links like exact timestamps, contiguous anomalous segments, and quantified severity measures — can make it reliable for offline data analysis. A principled evaluation framework for "insight quality" — combining claim-to-evidence traceability, correctness, and cost-aware exploration — remains underdeveloped compared to standard forecasting benchmarks.

### 2. Research object

#### 2.1 Object layer

Computer program

#### 2.2 Description

The research object **O** is a computer program: a set of time-series analysis tools, their structured output schemas, agent prompts, and harness plugins that mediate tool invocation and evidence-link extraction. This program sits between two fixed components.

The agent harness (upper layer) provides the planning/execution loop, tool-call routing, and budget enforcement. It is treated as fixed infrastructure — we do not modify it.

The LLM (lower layer) provides reasoning and natural-language generation capabilities. It is also treated as fixed — though fine-tuning may be explored in future work, it is out of scope for the current spec.

The research object consists of three interconnected components:
- **Tools**: time-series analysis functions (cheap scan, deep detect, decomposition, etc.) with precise I/O contracts
- **Output schemas**: structured formats that include evidence links (exact timestamps, contiguous segments, summary statistics)
- **Prompts and plugins**: agent instructions that constrain tool usage and harness plugins that enforce evidence-link extraction from tool outputs

#### 2.3 Assumptions / Restrictions

- Univariate time series only (multivariate excluded)
- Offline/batch analysis (streaming excluded)
- Agent communicates only through the tool interface (no direct raw data access)
- The harness and LLM are treated as fixed; modifications to either require a spec revision

### 3. Knowledge gap

**Gap 1: Tool composition for time series analysis.** Individual time-series tools (anomaly detection, change point detection, decomposition, forecasting) are well-studied, but there is no established framework for composing them into a budget-aware analysis workflow. Fixed pipelines (run everything on everything, or run a fixed sequence) waste compute on uninteresting series and cannot adapt tool selection per series.

**Gap 2: Evidence-grounded LLM agents for data analysis.** LLM-based agents have been applied to software engineering and web tasks, but not to time series orchestration where the agent has no direct data access and must rely entirely on structured tool outputs. It is unknown whether such agents can produce faithful, traceable narratives under a tool-call budget, and how to constrain them to avoid hallucination.

**Gap 3: Evaluation of insight quality vs. point accuracy.** Time series benchmarks measure point accuracy (MAE, RMSE, F1) but lack metrics for "insight quality" — the ability to surface what matters in a large pool, explain why with evidence links, and do so within a bounded cost budget.

**Gap 4: Context-efficient time series access.** When given raw data access, LLM agents retrieve and prefill full-length time series into context, wasting tokens on data that could be summarized or filtered upstream. There is no established pattern for structuring tool outputs so the agent can analyze series without consuming proportional context budget.

### 4. Research questions

**RQ1:** Can an LLM agent equipped with time-series-specific analysis tools (anomaly detection, decomposition, change point detection, etc.) outperform an agent that only has a tool for retrieving raw series data in analysis quality on large pools of univariate time series? By analysis quality we mean: correctness of the answer and evidence-grounded claims in the explanation.

**RQ2:** What metrics and evaluation protocols capture analysis quality — claim-to-evidence traceability, correctness, and budget efficiency?

**RQ3:** What tool output structure (summary statistics, evidence-linked segments, etc.) enables an LLM agent to analyze time series effectively without consuming context proportional to series length?

### 5. Research value

Answering these RQs is non-trivial because:

- **No established comparison exists.** LLM agents have not been benchmarked on time series analysis with controlled tool access (tool-augmented vs raw-retrieval-only).
- **No metric for evidence-grounded analysis quality.** Current benchmarks measure point accuracy, not claim-to-evidence traceability.
- **Empirical overhead is large.** Running the comparison requires building a full tool suite, harness integration, evaluation harness, and multi-dataset setup.
- **Cost scaling is unmeasured.** Processing large pools at multiple budgets with LLM agents has no published baseline for reasonable expectations.

### 6. Study type

Empirical characterization. The program is built as implementation; the study itself discovers the agent's properties through controlled experiments: comparative runs against baselines (RQ1), metric validation (RQ2), and context-efficiency measurements (RQ3). Each RQ maps to a concrete empirical question with measurable outcomes.

### 7. Resources & constraints

- **LLM API access**: up to 20B-parameter models, approximately 1M tokens per day
- **GPU**: 1× RTX 3090
- **Time**: 9 months

### 8. Validation philosophy

A result is valid if:
- **RQ1 (tool advantage):** Tool-augmented agent shows statistically significant improvement over raw-retrieval-only agent on selected datasets across analysis quality metrics (answer correctness + evidence-grounded explanation).
- **RQ2 (metrics):** Proposed metrics correlate with human judgments of analysis quality on a held-out evaluation set.
- **RQ3 (context efficiency):** Tool output structures yield comparable analysis quality while reducing context token consumption by at least 50% compared to raw-value access.

### 9. Roadmap

#### 9.1 Milestones

| ID | Name | Expected result | Time index | Strong scaling efficiency |
|----|------|-----------------|------------|---------------------------|
| M1 | Baseline development + dataset preparation | Raw-retrieval-only baseline agent (single tool providing time series access), dataset pool builder, label derivation for selected datasets | T+2 | 0.65 |
| M2 | Metrics + tool set → draft agent | Analysis quality metric suite (RQ2), complete tool set with evidence-linked output schemas, working draft agent | T+6 | 0.45 |
| M3 | Validation + context efficiency + agent polishing | Comparative validation results (RQ1), context-efficiency measurements (RQ3), polished agent incorporating feedback | T+9 | 0.55 |

#### 9.2 Long-term vision

- Transfer to other domains (finance, medicine, web metrics)
- Fine-tuning the LLM for time-series analysis tasks
- Extending to multivariate time series
- Real-time/streaming agent
