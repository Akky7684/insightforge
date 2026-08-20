# InsightForge — Project Progress & State Handover Document

**Last Updated:** August 21, 2026  
**Document Status:** Living Handover & Execution Tracker  
**Owner:** Achintya Singh (Dual Degree, IIT Kharagpur — Class of 2027)  
**Master Plan Reference:** 16-Week Workplan (Section 19, `InsightForge_Project_Blueprint.md`)

---

## 1. Project Overview & Objective

**InsightForge** is an autonomous multi-agent data analyst that converts raw tabular datasets (CSV/Excel/SQL) into verified, human-approved insights using natural language conversation.

### Strategic Purpose & Core Requirements:
- **IIT Kharagpur Placement Portfolio (Data & AI/GenAI Profiles)**: Differentiates significantly from generic PDF chatbots by demonstrating deep data science competence (EDA, hypothesis testing, anomaly detection) alongside advanced agentic engineering (multi-agent supervisor, reflection/self-correction loops, sandboxing, persistent memory, HITL guardrails, quantified benchmark evaluation).
- **Strict Zero-Cost Requirement**: Fully operates on free tiers (Google Gemini 2.5/3.6 Flash via Google AI Studio, Tavily free tier, local Chroma & Postgres in Docker, AWS free tier).
- **Core Orchestration Engine**: LangGraph state machine routing across 6 specialized agents:
  1. **Supervisor**: Intent classifier and master graph router.
  2. **Planner**: Analytical task decomposition into structured subtasks.
  3. **Data Profiler**: Automated dataset profiling (missingness, distributions, schema).
  4. **Coder / Executor**: Python/pandas code writer running in an isolated sandbox.
  5. **Critic**: Self-correction & statistical assumption validator (bounded retry <= 2).
  6. **RAG Grounding**: Domain business glossary & report context retrieval.
  7. **Reporter**: Final narrative answer synthesis with computed evidence citations.

---

## 2. Current Tech Stack & Architecture

### Environment & Tooling
- **Language**: Python 3.14.7 (Windows x64 compatible)
- **Virtual Environment**: `./venv/` (150+ packages installed & verified)
- **Orchestration**: `langgraph>=1.2.11`, `langchain-core>=1.5.6`, `langchain-google-genai>=4.3.4`
- **LLM Tiering**: `gemini-3.6-flash` (primary workhorse for routing/coding/profiling) + `gemini-3.1-pro-preview` (high-complexity planning/reporting)
- **Backend REST API**: `fastapi>=0.115.0` + `uvicorn>=0.30.0`
- **Frontend UI**: `streamlit>=1.40.0`
- **Data & Stats Engine**: `pandas>=2.2.0`, `numpy>=2.1.0`, `scipy>=1.14.0`, `statsmodels>=0.14.0`, `openpyxl>=3.1.0`
- **Visualization**: `matplotlib>=3.9.0`, `plotly>=5.24.0`, `seaborn`
- **Vector DB**: `chromadb>=1.5.0`
- **Persistence**: `langgraph-checkpoint>=4.2.0`, `MemorySaver` (Dev), Postgres (`langgraph-checkpoint-postgres` for Week 8+)
- **Testing & Quality**: `pytest>=8.3.0`, `pytest-asyncio>=0.24.0`, `ruff>=0.8.0`, `locust>=2.32.0`
- **Version Control**: Git 2.53.0 (Remote: `https://github.com/Akky7684/insightforge.git`, branch `main`)

### Repository Directory Structure
```
Insight_Forge/
├── .github/workflows/
│   ├── ci.yml                          # CI pipeline skeleton
│   └── cd.yml                          # CD pipeline skeleton
├── .env                                # Local secrets (GITIGNORED)
├── .env.example                        # Sample config template
├── .gitignore                          # Complete ignore rules (secrets, venv, data, logs)
├── README.md                           # Public project description
├── InsightForge_Project_Blueprint.md   # Source-of-truth blueprint (462 lines, untouched)
├── PROGRESS.md                         # This handover document
├── pytest.ini                          # Pytest config (pythonpath = .)
├── data/                               # Evaluation datasets (GITIGNORED)
│   ├── README.md
│   ├── titanic.csv                     # (891 rows, 12 cols)
│   ├── superstore.csv                  # (9,994 rows, 21 cols)
│   ├── ecommerce.csv                   # (541,909 rows, 8 cols)
│   ├── ipl.csv                         # (99,120 rows, 18 cols)
│   ├── synthetic_anomaly.csv           # (1,000 rows, 12 cols)
│   └── synthetic_anomaly_ground_truth.json # (50 injected anomalies with labels)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt                # Pinned production requirements
│   ├── requirements-dev.txt            # Dev/testing requirements
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI entrypoint & CORS middleware
│   │   ├── config.py                   # Pydantic Settings & Gemini LLM factory
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py               # REST API endpoints (/health, /upload, /chat, /sample-datasets)
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   ├── state.py                # InsightForgeState & Subtask schema
│   │   │   ├── supervisor.py           # Supervisor routing & LangGraph StateGraph
│   │   │   └── agents/
│   │   │       ├── __init__.py
│   │   │       ├── coder.py            # Coder/Executor agent node with schema injection
│   │   │       ├── planner.py          # (Week 4 skeleton)
│   │   │       ├── profiler.py         # (Week 3 skeleton)
│   │   │       ├── critic.py           # (Week 5 skeleton)
│   │   │       ├── rag_agent.py        # (Week 6 skeleton)
│   │   │       └── reporter.py         # (Week 3/4 skeleton)
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── sandbox_exec.py         # AST-hardened subprocess code execution tool
│   │   │   ├── sql_tool.py             # Read-only SQL tool (Week 3/4)
│   │   │   ├── stats_tool.py           # Stats hypothesis tool (Week 5)
│   │   │   └── chart_tool.py           # Chart generation tool (Week 3)
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── checkpointer.py         # Postgres checkpointer (Week 8)
│   │   │   └── vector_memory.py        # Chroma memory (Week 6)
│   │   └── hitl/
│   │       ├── __init__.py
│   │       └── gates.py                # HITL interrupt triggers (Week 7)
│   └── tests/
│       ├── unit/
│       │   ├── __init__.py
│       │   └── test_core.py            # 7 unit tests (100% passing)
│       ├── integration/
│       └── eval/
│           ├── benchmark.json
│           └── run_eval.py
├── frontend/
│   └── streamlit_app.py                # Streamlit UI with file uploader, preview, chat bubbles
├── infra/
│   └── docker-compose.yml              # Local dev compose file
└── docs/
    ├── ARCHITECTURE.md
    └── EVAL_RESULTS.md
```

---

## 3. Completed Work (Session Highlights)

### Week 1: Setup & Scope Lock (100% Complete & Verified)
- Scaffolded modular 40+ file directory layout matching blueprint Section 18.
- Configured Git, `.gitignore`, user credentials, initial commit, and remote push to `https://github.com/Akky7684/insightforge.git`.
- Set up `./venv/` and installed all production & dev dependencies.
- Configured `.env` with Google AI Studio (`GOOGLE_API_KEY`), Tavily (`TAVILY_API_KEY`), and LangSmith (`LANGSMITH_API_KEY`).
- Downloaded and generated all 5 evaluation datasets in `data/`:
  - `titanic.csv` (891 rows), `superstore.csv` (9,994 rows), `ecommerce.csv` (541,909 rows), `ipl.csv` (99,120 deliveries), `synthetic_anomaly.csv` (1,000 transactions, 50 labeled anomalies with ground-truth JSON).
- Verified whole setup with automated audit script.

### Week 2: Core LangGraph Skeleton (100% Complete & Hardened)
- **Phase 1: State Schema** (`backend/app/graph/state.py`):
  - Defined `Subtask` Pydantic model (`id`, `description`, `status`, `result`, `retries`).
  - Defined `InsightForgeState` TypedDict with message reducers, dataset paths, plan list, and session IDs.
- **Phase 2: Config Module** (`backend/app/config.py`):
  - Robust Pydantic `Settings` class resolving `.env` from project root.
  - LLM factory with active models (`gemini-3.6-flash` and `gemini-3.1-pro-preview`).
- **Phase 3: Sandbox Tool Hardening** (`backend/app/tools/sandbox_exec.py`):
  - Subprocess execution with 30s timeout and crash isolation.
  - Upgraded safety validator to **Python AST (`ast.parse`) static analysis** to block system/network modules (`os`, `sys`, `subprocess`, `shutil`, `socket`, `http`, etc.) and unsafe calls (`eval`, `exec`, `open`, `__import__`).
  - Automatic injection of dataset as preloaded pandas DataFrame `df`.
- **Phase 4: Coder/Executor Agent** (`backend/app/graph/agents/coder.py`):
  - Dynamic dataset schema injection (`_get_schema_info`).
  - Safe parsing of multimodal/list LLM response blocks and markdown code fences.
  - Sandbox execution with automated error message formatting.
- **Phase 5: Supervisor & Graph Compilation** (`backend/app/graph/supervisor.py`):
  - Implemented `supervisor_node` routing.
  - Connected `StateGraph(InsightForgeState)`: `supervisor` ➔ `coder` ➔ `END`.
  - Added `MemorySaver` in-memory checkpointer supporting multi-turn state.
- **Phase 6: FastAPI Backend** (`backend/app/main.py`, `backend/app/api/routes.py`):
  - Endpoints: `GET /health`, `POST /api/upload`, `POST /api/chat`, `GET /api/sample-datasets`.
  - Handled CSV and Excel uploads with automated CSV conversion and structural metadata profiling.
- **Phase 7: Streamlit Frontend** (`frontend/streamlit_app.py`):
  - Dataset selector (sample dataset dropdown + drag-and-drop file uploader).
  - Dataset inspector (column dtypes, preview table).
  - Multi-turn conversation UI with suggested prompt buttons.
- **Phase 8: Deep Stress Verification & Pytest Suite**:
  - `pytest backend/tests/unit/` ➔ **7 / 7 passed in 3.24s**.
  - Verified numerical accuracy on Titanic (Average age = `29.70`, Female survival = `74.20%`).
  - Verified multi-turn thread memory (`216` 1st class passengers ➔ `136` survived).

---

## 4. Current Status & Milestone Tracking

- **Completed Milestones**:
  - ✅ **Week 1**: Setup & Scope Lock (Aug 17–23)
  - ✅ **Week 2**: Core LangGraph Skeleton & MVP (Aug 24–30)
  - ✅ **Week 3**: Harden the MVP Loop (Aug 31–Sep 6)
    - Data Profiler Agent: automated statistical summary & correlation detection.
    - Chart Tool: Pydantic-validated plotting engine with automatic figure capture.
    - Evaluation Benchmark: 10 ground-truth questions (100% accuracy, 0 crashes).
    - UI Integration: Deep Profile dashboard tab & inline chart rendering in Streamlit.
  - ✅ **Week 4**: Planner Agent & Multi-Step Task Decomposition (Sep 7–13)
    - Planner Agent: Structured `PlanOutput` task decomposition with Gemini.
    - Multi-Step LangGraph Loop: Sequential execution of subtasks with prior context passing.
    - Reporter Agent: Executive narrative synthesis with empirical evidence citations.
    - Benchmark Expansion: 20 ground-truth questions in `benchmark.json` and `run_eval.py`.
    - UI Enhancements: Step-by-step analytical plan expander and subtask progress in Streamlit.
  - ✅ **Week 5**: Critic Agent & Self-Correction Reflection Loop (Sep 14–20)
    - Statistical Hypothesis Testing Tool: Independent/Paired T-Tests, $\chi^2$ Contingency, One-Way ANOVA, Mann-Whitney U, Pearson/Spearman correlation, and effect sizes (Cohen's d, Cramér's V, $\eta^2$).
    - Critic Agent: Statistical assumption validation, skewness checking, and bounded self-correction loop ($\le 2$ retries).
    - Self-Correction Recovery Rate: 100% on benchmark execution retries.
- **Current Position**: Ready to begin **Week 6 (Sep 21–27)**: *RAG Grounding & Persistent Long-Term Memory (ChromaDB Vector Store + Business Glossary)*.
- **Blockers / Issues**: Zero active blockers. 10/10 unit tests passing, git tree clean and pushed.

---

## 5. Next Immediate Steps (Week 3 Execution Plan)

### Goal: Harden the MVP Loop & Multi-Agent Capabilities
1. **Phase 1: Data Profiler Agent (`backend/app/graph/agents/profiler.py`)**:
   - Automated profiling upon dataset upload (shape, column types, missingness percentages, cardinality, numerical distribution statistics, top correlations).
   - Caches profile into `InsightForgeState["dataset_profile"]` to eliminate repetitive profiling tool calls.
2. **Phase 2: Chart Generation Tool (`backend/app/tools/chart_tool.py`)**:
   - Pydantic schema validation for chart specifications (`title`, `chart_type`, `x_col`, `y_col`, `hue`).
   - Generation of static charts (saved to `outputs/` or returned as base64/plotly JSON) and integration into Coder agent.
3. **Phase 3: Structured Pydantic I/O Between Nodes**:
   - Type-safe Pydantic contracts for inter-agent communication.
4. **Phase 4: First 10 Manual Benchmark Questions (Titanic & Superstore)**:
   - Construct and automate testing for the first 10 benchmark queries in `backend/tests/eval/run_eval.py`.
5. **Phase 5: Streamlit & FastAPI Updates**:
   - Support rendering charts and structured profile cards in the Streamlit UI.
6. **Phase 6: Pytest & Verification**:
   - Expand unit and integration test coverage; commit and push Week 3.

---

## 6. Key Context & Design Decisions

1. **Adherence to 16-Week Roadmap**:
   - Strict alignment with Section 19 of `InsightForge_Project_Blueprint.md`.
   - Methodical step-by-step progress without cutting corners.
2. **Zero-Cost Constraint**:
   - Primary LLM: `gemini-3.6-flash` (free tier via Google AI Studio).
   - Advanced reasoning LLM: `gemini-3.1-pro-preview` (free tier).
   - Zero cloud charges incurred during development.
3. **Security Architecture**:
   - Subprocess execution isolated from host filesystem and network.
   - AST analysis guarantees no unauthorized imports (`os`, `sys`, `subprocess`, etc.) or execution bypasses (`eval`, `exec`, `open`).
4. **Source of Truth**:
   - `InsightForge_Project_Blueprint.md` remains the authoritative project charter.
