"""Unit tests for InsightForge Core Engine & Week 9 Automated EDA & Insights Agent."""

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from backend.app.config import get_settings
from backend.app.graph.agents.coder import _get_schema_info
from backend.app.graph.agents.critic import critic_node
from backend.app.graph.agents.eda_agent import generate_executive_eda
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
from backend.app.tools.eda_tool import generate_eda_report
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

    matches = search_glossary("What is the average order value formula?", top_k=1)
    assert len(matches) > 0
    assert "AOV" in matches[0]["term"] or "Order Value" in matches[0]["term"]

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


def test_automated_eda_engine():
    """Verify Automated EDA engine produces quality scores, insights, and 4-panel visual."""
    eda = generate_eda_report(TITANIC_PATH)
    assert eda["row_count"] == 891
    assert eda["column_count"] == 12
    assert eda["data_quality_score"] > 70.0
    assert len(eda["ranked_insights"]) >= 3
    assert eda["chart_path"] is not None
    assert Path(eda["chart_path"]).exists()


def test_automated_eda_agent_synthesis():
    """Verify EDA Agent synthesizes findings into an executive report."""
    briefing = generate_executive_eda(TITANIC_PATH)
    assert briefing["data_quality_score"] > 70.0
    assert len(briefing["narrative_report"]) > 100
    assert Path(briefing["chart_path"]).exists()


def test_anomaly_detection_engine():
    """Verify Anomaly Detection Engine runs algorithms and returns plot."""
    from backend.app.tools.anomaly_tool import detect_anomalies
    
    # Test Isolation Forest
    res_iso = detect_anomalies(SUPERSTORE_PATH, method="isolation_forest", contamination=0.05)
    assert res_iso["method"] == "isolation_forest"
    assert res_iso["total_anomalies"] > 0
    assert Path(res_iso["chart_path"]).exists()
    
    # Test Robust Z-Score
    res_z = detect_anomalies(SUPERSTORE_PATH, method="zscore", contamination=0.05)
    assert res_z["method"] == "zscore"
    assert res_z["total_anomalies"] > 0
    assert Path(res_z["chart_path"]).exists()


def test_anomaly_agent_narrative():
    """Verify Anomaly Agent synthesizes explanation for flagged rows."""
    from backend.app.graph.agents.anomaly_agent import generate_anomaly_report
    res = generate_anomaly_report(TITANIC_PATH, contamination=0.02)
    assert "narrative_report" in res
    assert len(res["narrative_report"]) > 50


def test_predictive_tool_classification():
    """Verify Auto-ML predictive engine trains classification models and ranks feature importance."""
    from backend.app.tools.predictive_tool import train_predictive_model
    res = train_predictive_model(TITANIC_PATH, target_column="Survived")
    assert res["task_type"] == "classification"
    assert res["metrics"]["accuracy_pct"] >= 70.0
    assert len(res["top_features"]) > 0
    assert Path(res["chart_path"]).exists()


def test_predictive_agent_synthesis():
    """Verify Predictive Agent synthesizes executive briefing for model."""
    from backend.app.graph.agents.predictive_agent import generate_predictive_report
    res = generate_predictive_report(TITANIC_PATH, target_column="Survived")
    assert "narrative_report" in res
    assert len(res["narrative_report"]) > 50
    assert Path(res["chart_path"]).exists()


def test_fastapi_endpoints_including_eda_and_predictive():
    """Verify FastAPI /health, /api/profile, /api/eda/generate, and /api/predictive/train endpoints."""
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    eda_res = client.post(f"/api/eda/generate?dataset_path={TITANIC_PATH}")
    assert eda_res.status_code == 200
    data = eda_res.json()
    assert data["row_count"] == 891
    assert len(data["ranked_insights"]) >= 3

    pred_res = client.post(f"/api/predictive/train?dataset_path={TITANIC_PATH}&target_column=Survived")
    assert pred_res.status_code == 200
    pdata = pred_res.json()
    assert pdata["task_type"] == "classification"
    assert pdata["metrics"]["accuracy_pct"] >= 70.0
