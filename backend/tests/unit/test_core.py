"""Unit tests for InsightForge Core Engine & Week 6 RAG Vector Memory."""

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from backend.app.config import get_settings
from backend.app.graph.agents.coder import _get_schema_info
from backend.app.graph.agents.critic import critic_node
from backend.app.graph.agents.planner import PlanOutput, planner_node
from backend.app.graph.agents.profiler import profile_dataset
from backend.app.graph.agents.rag_agent import rag_node
from backend.app.graph.agents.reporter import reporter_node
from backend.app.graph.state import Subtask
from backend.app.graph.supervisor import get_graph, supervisor_node
from backend.app.main import app
from backend.app.memory.vector_memory import (
    add_glossary_term,
    list_all_glossary_terms,
    save_analysis,
    search_glossary,
    search_past_analyses,
)
from backend.app.tools.chart_tool import ChartSpec, render_chart_from_spec
from backend.app.tools.sandbox_exec import execute_code_in_sandbox
from backend.app.tools.stats_tool import StatsTestRequest, run_stats_test

DATA_DIR = Path(get_settings().data_dir)
TITANIC_PATH = str((DATA_DIR / "titanic.csv").resolve())
SUPERSTORE_PATH = str((DATA_DIR / "superstore.csv").resolve())


def test_settings_load():
    """Verify settings load properly with valid data directory."""
    settings = get_settings()
    assert "flash" in settings.gemini_flash_model
    assert Path(settings.data_dir).exists()


def test_subtask_model():
    """Verify Subtask Pydantic model serialization and validation."""
    subtask = Subtask(id="task_01", description="Calculate mean fare")
    assert subtask.status == "pending"
    assert subtask.retries == 0
    data = subtask.model_dump()
    assert data["id"] == "task_01"


def test_sandbox_safe_execution():
    """Verify sandbox runs standard pandas calculations correctly."""
    res = execute_code_in_sandbox("print(round(df['Fare'].mean(), 2))", dataset_path=TITANIC_PATH)
    assert res["success"] is True
    assert res["error"] is None
    assert "32.2" in res["output"]


def test_sandbox_security_blocks_dangerous_imports():
    """Verify AST sandbox rejects unauthorized system imports."""
    res_os = execute_code_in_sandbox("import os\nprint(os.getcwd())", dataset_path=TITANIC_PATH)
    assert res_os["success"] is False
    assert "Blocked import" in res_os["error"]

    res_sub = execute_code_in_sandbox("import subprocess\nsubprocess.run(['dir'])", dataset_path=TITANIC_PATH)
    assert res_sub["success"] is False
    assert "Blocked import" in res_sub["error"]


def test_data_profiler_agent():
    """Verify Data Profiler computes accurate missingness, stats, and correlations."""
    prof = profile_dataset(TITANIC_PATH)
    assert prof["row_count"] == 891
    assert prof["column_count"] == 12
    assert "Age" in prof["numerical_stats"]
    assert prof["numerical_stats"]["Age"]["mean"] == 29.7
    assert prof["numerical_stats"]["Age"]["outlier_count"] == 11
    assert len(prof["high_correlations"]) >= 1


def test_chart_generation_tool():
    """Verify chart tool renders and outputs a valid PNG image file."""
    df = pd.read_csv(TITANIC_PATH, encoding="latin1")
    spec = ChartSpec(
        chart_type="bar",
        x="Pclass",
        y="Survived",
        aggregation="mean",
        title="Class Survival Test",
    )
    res = render_chart_from_spec(spec, df)
    assert res["success"] is True
    assert res["chart_path"] is not None
    assert Path(res["chart_path"]).exists()
    assert len(res["base64"]) > 100


def test_stats_tool_hypothesis_testing():
    """Verify statistical hypothesis testing engine on Titanic dataset."""
    df = pd.read_csv(TITANIC_PATH, encoding="latin1")

    # Chi-Square Test
    chi2_res = run_stats_test(df, StatsTestRequest(test_type="chi2_contingency", var1="Sex", var2="Survived"))
    assert chi2_res.is_significant is True
    assert chi2_res.p_value < 0.001
    assert chi2_res.effect_size["metric"] == "Cramér's V"

    # ANOVA Test
    anova_res = run_stats_test(df, StatsTestRequest(test_type="anova", var1="Fare", group_col="Pclass"))
    assert anova_res.is_significant is True
    assert anova_res.statistic > 100.0


def test_vector_memory_glossary_crud_and_search():
    """Verify ChromaDB business glossary storage and semantic similarity search."""
    terms = list_all_glossary_terms()
    assert len(terms) >= 3

    # Semantic search for AOV
    matches = search_glossary("What is the average order value formula?", top_k=1)
    assert len(matches) > 0
    assert "AOV" in matches[0]["term"] or "Order Value" in matches[0]["term"]
    assert matches[0]["similarity"] > 0.3

    # Add custom term and retrieve it
    t_id = add_glossary_term(
        term="Customer Retention Rate",
        definition="Proportion of active users retained over a period.",
        formula="(End Customers - New Customers) / Start Customers * 100",
        category="Growth",
    )
    assert t_id.startswith("custom_")

    # Index and search past analysis
    a_id = save_analysis(
        query="Calculate AOV in 2017",
        dataset_name="superstore.csv",
        code="df['Sales'].sum() / df['Order ID'].nunique()",
        report_summary="AOV was $458.60.",
    )
    assert a_id.startswith("analysis_")
    past_matches = search_past_analyses("AOV query", dataset_name="superstore.csv", top_k=1)
    assert len(past_matches) > 0


def test_rag_agent_grounding_node():
    """Verify RAG Grounding node injects domain formulas into state context."""
    state = {
        "messages": [HumanMessage(content="Calculate the average order value (AOV) for our store.")],
        "dataset_id": "superstore.csv",
    }
    rag_res = rag_node(state)
    assert rag_res["rag_context"] is not None
    assert "Average Order Value" in rag_res["rag_context"] or "AOV" in rag_res["rag_context"]


def test_critic_agent_reflection_retry():
    """Verify Critic agent catches runtime execution failure and increments retries."""
    flawed_plan = [
        Subtask(id="task_1", description="Calculate mean age", status="failed", result="Error during execution: NameError: name 'age' is not defined", retries=0)
    ]
    state = {
        "messages": [HumanMessage(content="What is the average age?")],
        "plan": flawed_plan,
        "current_subtask_idx": 1,
    }
    cmd = critic_node(state)
    assert cmd.goto == "coder"
    assert cmd.update["plan"][0].retries == 1
    assert "RETRY 1" in cmd.update["plan"][0].description


def test_reporter_agent_synthesis():
    """Verify reporter agent combines executed subtask outputs cleanly."""
    plan = [
        Subtask(id="task_1", description="Calculate mean age", status="success", result="Mean age: 29.7"),
        Subtask(id="task_2", description="Calculate survival %", status="success", result="Survival: 38.38%"),
    ]
    state = {
        "messages": [HumanMessage(content="Summarize age and survival stats.")],
        "plan": plan,
    }
    cmd = reporter_node(state)
    report_text = cmd["messages"][0].content
    assert "29.7" in report_text or "38.38" in report_text


def test_fastapi_endpoints():
    """Verify FastAPI /health, /api/sample-datasets, and /api/profile endpoints."""
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    datasets = client.get("/api/sample-datasets")
    assert datasets.status_code == 200
    names = [d["name"] for d in datasets.json()]
    assert "titanic.csv" in names

    profile_res = client.get(f"/api/profile?dataset_path={TITANIC_PATH}")
    assert profile_res.status_code == 200
    assert profile_res.json()["row_count"] == 891
