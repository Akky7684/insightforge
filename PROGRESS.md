# InsightForge — Project Progress & State Handover Document

**Last Updated:** August 21, 2026  
**Document Status:** Living Handover / Progress Tracker  
**Owner:** Achintya Singh (Dual Degree, IIT Kharagpur — Class of 2027)

---

## 1. Project Overview & Objective

**InsightForge** is an autonomous multi-agent data analyst that converts tabular datasets (CSV/Excel/SQL) into verified, human-approved insights using natural language conversation.

### Core Objectives & Highlights
- **Multi-Agent Orchestration**: Supervisor routes tasks across 6 specialized agents (Supervisor, Planner, Data Profiler, Coder/Executor, Critic, RAG Grounding, Reporter).
- **Self-Correction / Reflection**: Critic agent validates execution & statistical assumptions with bounded retry loops (<= 2 retries).
- **Human-in-the-Loop (HITL)**: Interrupt gates for risky operations (destructive schema changes, expensive scans, data export, assumption violations).
- **Sandboxed Execution**: Subprocess-isolated code execution with strict import whitelisting (blocking `os`, `sys`, `subprocess`, etc.) and timeouts.
- **Evaluation Harness**: 90-question benchmark across 5 datasets with quantified accuracy, latency, cost, and recovery rate metrics.
- **Persistence & Observability**: Postgres checkpoints (`langgraph-checkpoint-postgres`), Chroma vector store for long-term memory, LangSmith tracing.
- **Zero-Cost Strategy**: Utilizes Google Gemini 2.5 Flash / Pro via Google AI Studio free tier, Tavily free tier, local Chroma & Postgres in Docker, AWS free tier for deployment.

---

## 2. Current Tech Stack & Architecture

### Environment & Tooling
- **Language**: Python 3.14.7 (Windows x64)
- **Virtual Environment**: `./venv/` (contains 150+ installed dependencies)
- **Orchestration**: LangGraph 1.2.11, LangChain Core 1.5.6, LangChain Google GenAI 4.3.4
- **Web Framework**: FastAPI 0.141.1 + Uvicorn 0.52.3
- **Frontend**: Streamlit 1.61.1
- **Data & Stats**: Pandas 3.0.5, NumPy 2.5.2, SciPy 1.18.0, Statsmodels 0.14.6
- **Visualization**: Matplotlib 3.11.1, Plotly 6.9.0
- **Vector DB**: ChromaDB 1.5.9
- **Database**: PostgreSQL driver (`psycopg2-binary`, `psycopg-pool`, `langgraph-checkpoint-postgres`)
- **Version Control**: Git 2.53.0 (Remote: `https://github.com/Akky7684/insightforge.git`, branch `main`)
- **Containerization**: Docker 29.3.1

### Directory Structure Established
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
│   │   ├── main.py                     # FastAPI entrypoint (skeleton)
│   │   ├── config.py                   # Pydantic Settings & Gemini LLM factory (COMPLETED)
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py               # REST API endpoints (skeleton)
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   ├── state.py                # InsightForgeState & Subtask schema (COMPLETED)
│   │   │   ├── supervisor.py           # Supervisor routing & LangGraph StateGraph (COMPLETED)
│   │   │   └── agents/
│   │   │       ├── __init__.py
│   │   │       ├── coder.py            # Coder/Executor agent node (COMPLETED)
│   │   │       ├── planner.py          # (Week 4 skeleton)
│   │   │       ├── profiler.py         # (Week 3 skeleton)
│   │   │       ├── critic.py           # (Week 5 skeleton)
│   │   │       ├── rag_agent.py        # (Week 6 skeleton)
│   │   │       └── reporter.py         # (Week 3/4 skeleton)
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── sandbox_exec.py         # Subprocess code execution tool (COMPLETED)
│   │   │   ├── sql_tool.py             # Read-only SQL tool (skeleton)
│   │   │   ├── stats_tool.py           # Stats hypothesis tool (skeleton)
│   │   │   └── chart_tool.py           # Chart generation tool (skeleton)
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── checkpointer.py         # Postgres checkpointer (Week 8)
│   │   │   └── vector_memory.py        # Chroma memory (Week 6)
│   │   └── hitl/
│   │       ├── __init__.py
│   │       └── gates.py                # HITL interrupt triggers (Week 7)
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── eval/
│           ├── benchmark.json
│           └── run_eval.py
├── frontend/
│   └── streamlit_app.py                # Streamlit UI
├── infra/
│   └── docker-compose.yml              # Local dev compose file
└── docs/
    ├── ARCHITECTURE.md
    └── EVAL_RESULTS.md
```

---

## 3. Completed Work (Session Highlights)

### Week 1: Setup & Scope Lock (100% Complete & Pushed)
- **Phase 1: Repo Scaffolding & Git Init**
  - Scaffolded full 40+ file modular repository structure according to blueprint Section 18.
  - Initialized Git, created clean `.gitignore` (ignoring `.env`, `.csv`, `venv`, logs, cache).
  - Configured git user (`Achintya Singh <achintyasingh684@gmail.com>`).
  - Created initial commit and connected remote repository `https://github.com/Akky7684/insightforge.git`.
  - Pushed `main` branch to GitHub.
- **Phase 2: Python Environment & Dependencies**
  - Verified Python 3.14.7 compatibility across all core packages.
  - Created virtual environment `./venv/`.
  - Created `backend/requirements.txt` and `backend/requirements-dev.txt`.
  - Installed all 150+ packages cleanly. Verified all core imports.
- **Phase 3: API Keys & Secrets Configuration**
  - Created `.env` and `.env.example`.
  - Configured Google AI Studio API key (`GOOGLE_API_KEY`), Tavily key (`TAVILY_API_KEY`), and LangSmith key (`LANGSMITH_API_KEY`).
  - Verified non-exposure and presence of active API keys via script.
- **Phase 4: Dataset Acquisition & Generation**
  - Acquired `titanic.csv` (891 rows).
  - Acquired `superstore.csv` (9,994 rows).
  - Acquired `ecommerce.csv` (541,909 rows).
  - Generated realistic ball-by-ball cricket dataset `ipl.csv` (99,120 deliveries).
  - Generated `synthetic_anomaly.csv` (1,000 transactions) with 5 distinct anomaly types and saved exact ground truth to `synthetic_anomaly_ground_truth.json`.
- **Phase 5: Verification & Push**
  - Executed end-to-end verification script testing imports, dataset integrity, environment security, and git status.
  - Committed and pushed Week 1 milestone to GitHub.

---

### Week 2: Core LangGraph Skeleton (In Progress ~60% Complete)
- **Phase 1: State Schema** (`backend/app/graph/state.py`):
  - Created `Subtask` Pydantic model with fields (`id`, `description`, `status`, `result`, `retries`).
  - Created `InsightForgeState` TypedDict with messages reducer, dataset fields (`dataset_id`, `dataset_path`, `dataset_profile`), plan list, RAG context, and session info.
- **Phase 2: Config Module** (`backend/app/config.py`):
  - Created `Settings` Pydantic BaseSettings class loading `.env` relative to project root.
  - Added `get_llm(model_type='flash'|'pro')` factory with deterministic temperature (0.0).
- **Phase 3: Sandbox Execution Tool** (`backend/app/tools/sandbox_exec.py`):
  - Built isolated subprocess execution script runner with stdout/stderr capture and 30s timeout.
  - Implemented strict safety filter blocking dangerous imports (`os`, `sys`, `subprocess`, `shutil`, `pathlib`, `socket`, `http`, etc.).
  - Added automatic dataframe injection (`df = pd.read_csv(...)`).
  - Wrapped as `@tool sandbox_exec`.
  - Verified across 5 test cases (standard pandas queries, groupbys, blocked imports, runtime errors).
- **Phase 4: Coder/Executor Agent** (`backend/app/graph/agents/coder.py`):
  - Designed system prompt for expert pandas analyst with dynamic schema injection (`_get_schema_info`).
  - Integrated code extraction (stripping markdown fences) and sandbox execution.
- **Phase 5: Supervisor + Graph Wiring** (`backend/app/graph/supervisor.py`):
  - Built `supervisor_node` routing logic and dataset validation.
  - Constructed `StateGraph(InsightForgeState)` connecting `supervisor` -> `coder` -> `END`.
  - Integrated `MemorySaver` in-memory checkpointer.

---

## 4. Current Status & In-Flight Task

**Current Phase**: Week 2, Phase 5 Verification -> Phase 6 (FastAPI Backend)

We have just created `coder.py` and `supervisor.py`. 
Next action:
1. Run an end-to-end test of the compiled graph on `titanic.csv` ("What is the average age of passengers?").
2. Implement **Phase 6: FastAPI Backend** (`backend/app/main.py`, `backend/app/api/routes.py`).
3. Implement **Phase 7: Streamlit Frontend** (`frontend/streamlit_app.py`).
4. Implement **Phase 8: Integration Testing, Polish, Commit & Push**.

---

## 5. Next Immediate Steps (Step-by-Step Checklist)

- [ ] **Step 1: Test LangGraph End-to-End**:
  - Run a quick test script invoking `get_graph()` with sample message on `titanic.csv`.
  - Verify Gemini Flash generates pandas code and sandbox executes returning ~29.7.
- [ ] **Step 2: Build FastAPI Backend** (Phase 6):
  - Update `backend/app/api/routes.py`:
    - `POST /api/upload`: handles CSV/Excel upload, saves to `./uploads/`, generates schema info.
    - `POST /api/chat`: accepts query + `session_id` + `dataset_path`, runs graph, returns response stream or JSON.
  - Update `backend/app/main.py`: configure CORS, mount routes, add `/health`.
- [ ] **Step 3: Build Streamlit Frontend** (Phase 7):
  - Update `frontend/streamlit_app.py`:
    - Sidebar: dataset uploader & preview (shape, columns, head).
    - Main chat interface: multi-turn message history with chat bubbles.
    - Connect directly or via FastAPI client.
- [ ] **Step 4: Integration Test & Verification** (Phase 8):
  - Launch FastAPI + Streamlit.
  - Test asking:
    - "What is the survival rate by gender?"
    - "How many passengers were in 1st class?"
    - "What is the average fare for survivors vs non-survivors?"
- [ ] **Step 5: Git Commit & Push Week 2**:
  - Stage all files, commit: `Week 2 complete — LangGraph MVP with Supervisor, Coder, FastAPI, and Streamlit UI`.
  - Push to GitHub origin `main`.

---

## 6. Key Context & Design Decisions

1. **Strict No-Cost Policy**:
   - Zero budget constraint: using Gemini 2.5 Flash as the primary workhorse, with Gemini 2.5 Pro reserved only for high-complexity tasks.
   - All APIs used are on free tiers.
2. **Windows Compatibility**:
   - Subprocess sandboxing uses standard Python timeouts without Linux-specific `resource.setrlimit`. (Full container isolation scheduled for AWS deployment phase).
3. **Pace & Governance**:
   - Proceed methodically week-by-week and phase-by-phase without rushing.
   - Every phase verified before proceeding to the next.
4. **Source of Truth**:
   - `InsightForge_Project_Blueprint.md` remains the authoritative specification for all components.
