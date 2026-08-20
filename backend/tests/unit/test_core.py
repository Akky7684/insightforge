"""Unit tests for InsightForge Week 2 Core Engine."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from backend.app.config import get_settings
from backend.app.graph.agents.coder import _get_schema_info
from backend.app.graph.state import Subtask
from backend.app.graph.supervisor import get_graph, supervisor_node
from backend.app.main import app
from backend.app.tools.sandbox_exec import execute_code_in_sandbox

DATA_DIR = Path(get_settings().data_dir)
TITANIC_PATH = str((DATA_DIR / "titanic.csv").resolve())


def test_settings_load():
    """Verify settings load properly with valid data directory."""
    settings = get_settings()
    assert settings.gemini_flash_model == "gemini-3.6-flash"
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
    """Verify sandbox rejects unauthorized system imports."""
    res_os = execute_code_in_sandbox("import os\nprint(os.getcwd())", dataset_path=TITANIC_PATH)
    assert res_os["success"] is False
    assert "Blocked import" in res_os["error"]

    res_sub = execute_code_in_sandbox("import subprocess\nsubprocess.run(['dir'])", dataset_path=TITANIC_PATH)
    assert res_sub["success"] is False
    assert "Blocked import" in res_sub["error"]


def test_schema_info_generation():
    """Verify schema info string generator extracts column names and sample rows."""
    info = _get_schema_info(TITANIC_PATH)
    assert "Total rows: 891" in info
    assert "Survived" in info
    assert "Pclass" in info


def test_supervisor_routes_to_coder():
    """Verify supervisor routes valid input with dataset to coder."""
    state = {
        "messages": [HumanMessage(content="What is the average fare?")],
        "dataset_path": TITANIC_PATH,
    }
    cmd = supervisor_node(state)
    assert cmd.goto == "coder"


def test_fastapi_endpoints():
    """Verify FastAPI /health and /api/sample-datasets endpoints."""
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    datasets = client.get("/api/sample-datasets")
    assert datasets.status_code == 200
    names = [d["name"] for d in datasets.json()]
    assert "titanic.csv" in names
