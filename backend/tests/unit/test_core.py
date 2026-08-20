"""Unit tests for InsightForge Core Engine & Week 4 Multi-Agent Pipeline."""

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from backend.app.config import get_settings
from backend.app.graph.agents.coder import _get_schema_info
from backend.app.graph.agents.planner import PlanOutput, planner_node
from backend.app.graph.agents.profiler import profile_dataset
from backend.app.graph.agents.reporter import reporter_node
from backend.app.graph.state import Subtask
from backend.app.graph.supervisor import get_graph, supervisor_node
from backend.app.main import app
from backend.app.tools.chart_tool import ChartSpec, render_chart_from_spec
from backend.app.tools.sandbox_exec import execute_code_in_sandbox

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


def test_supervisor_routes_to_profiler_then_planner():
    """Verify supervisor routes to profiler first if profile missing, then to planner."""
    state_without_profile = {
        "messages": [HumanMessage(content="What is the average fare?")],
        "dataset_path": TITANIC_PATH,
        "dataset_profile": None,
    }
    cmd1 = supervisor_node(state_without_profile)
    assert cmd1.goto == "profiler"

    state_with_profile = {
        "messages": [HumanMessage(content="What is the average fare?")],
        "dataset_path": TITANIC_PATH,
        "dataset_profile": {"dataset_name": "titanic.csv"},
    }
    cmd2 = supervisor_node(state_with_profile)
    assert cmd2.goto == "planner"


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
