# InsightForge — Autonomous Multi-Agent Data Analyst

> A multi-agent system that turns raw datasets into verified, human-approved insights through natural-language conversation.

## Overview

InsightForge uses a **6-agent LangGraph orchestration system** (Supervisor, Planner, Coder/Executor, Critic, RAG Grounding, Reporter) to perform automated exploratory data analysis, hypothesis testing, and anomaly detection over arbitrary tabular datasets via natural language.

## Key Features

- **Multi-Agent Architecture** — Supervisor routes work across specialized agents
- **Self-Correction Loop** — Critic validates outputs with bounded retries
- **Human-in-the-Loop (HITL)** — Approval gating for destructive/high-cost operations
- **RAG Grounding** — Business glossary integration to reduce hallucinations
- **Sandboxed Code Execution** — Isolated subprocess with restricted imports
- **Persistent Memory** — Postgres-backed state + Chroma vector store
- **Evaluation Harness** — 90-question benchmark across 5 datasets with measured metrics
- **Full Observability** — LangSmith tracing with token/cost tracking

## Tech Stack

LangGraph · Gemini · FastAPI · Streamlit · PostgreSQL · ChromaDB · Docker · AWS · GitHub Actions

## Quickstart

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/insightforge.git
cd insightforge

# Start all services
docker-compose -f infra/docker-compose.yml up
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture deep-dive.

## Evaluation Results

See [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md) for benchmark results.

## Known Limitations

- Sandbox uses subprocess isolation, not full VM-level isolation (upgrade path: container-per-execution)
- Single-instance deployment, not HA (upgrade path: ECS Fargate + multi-AZ)
- Single-user demo auth only (upgrade path: multi-tenant auth)

## License

MIT
