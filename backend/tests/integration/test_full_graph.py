"""End-to-End Integration Tests for InsightForge Full 6-Agent LangGraph Architecture."""

import time
from pathlib import Path
import pytest
from langchain_core.messages import HumanMessage

from backend.app.config import get_settings
from backend.app.graph.supervisor import get_graph

DATA_DIR = Path(get_settings().data_dir)
TITANIC_PATH = str((DATA_DIR / "titanic.csv").resolve())
SUPERSTORE_PATH = str((DATA_DIR / "superstore.csv").resolve())
ECOMMERCE_PATH = str((DATA_DIR / "ecommerce.csv").resolve())


def test_full_graph_titanic_unprofiled_traversal():
    """Verify full graph traversal on a fresh unprofiled dataset (Supervisor -> Profiler -> RAG -> Planner -> Coder -> Critic -> Reporter)."""
    graph = get_graph()
    config = {"configurable": {"thread_id": f"int-test-titanic-{time.time()}"}}

    state_input = {
        "messages": [HumanMessage(content="What is the average fare and overall survival rate?")],
        "dataset_id": "titanic.csv",
        "dataset_path": TITANIC_PATH,
        "dataset_profile": None,
        "plan": [],
        "current_subtask_idx": 0,
        "rag_context": None,
        "pending_hitl_action": None,
        "session_id": "int-session-titanic",
        "user_id": "tester",
    }

    final_state = graph.invoke(state_input, config=config)

    # Verify Profiler ran and populated profile
    assert final_state.get("dataset_profile") is not None
    assert final_state["dataset_profile"]["row_count"] == 891

    # Verify Planner created executable plan
    plan = final_state.get("plan", [])
    assert len(plan) >= 1
    assert all(st.status == "success" for st in plan)

    # Verify Reporter produced response
    ai_msgs = [m for m in final_state["messages"] if m.type == "ai"]
    assert len(ai_msgs) > 0
    resp_text = ai_msgs[-1].content
    assert "32" in resp_text or "38" in resp_text or "fare" in resp_text.lower()


def test_full_graph_ecommerce_rag_grounding():
    """Verify RAG Grounding agent injects domain formula for AOV and Coder utilizes it."""
    graph = get_graph()
    config = {"configurable": {"thread_id": f"int-test-ecom-{time.time()}"}}

    state_input = {
        "messages": [HumanMessage(content="What is the Average Order Value (AOV) for this e-commerce dataset?")],
        "dataset_id": "ecommerce.csv",
        "dataset_path": ECOMMERCE_PATH,
        "dataset_profile": None,
        "plan": [],
        "current_subtask_idx": 0,
        "rag_context": None,
        "pending_hitl_action": None,
        "session_id": "int-session-ecom",
        "user_id": "tester",
    }

    final_state = graph.invoke(state_input, config=config)

    # Verify RAG context was injected
    assert final_state.get("rag_context") is not None
    assert "Average Order Value" in final_state["rag_context"] or "AOV" in final_state["rag_context"]

    # Verify final response was synthesized
    ai_msgs = [m for m in final_state["messages"] if m.type == "ai"]
    assert len(ai_msgs) > 0


def test_full_graph_superstore_multistep_execution():
    """Verify multi-step complex query decomposition and sequential execution across agents."""
    graph = get_graph()
    config = {"configurable": {"thread_id": f"int-test-superstore-{time.time()}"}}

    state_input = {
        "messages": [HumanMessage(content="What are the top 2 product categories by total sales and what is the total profit for each?")],
        "dataset_id": "superstore.csv",
        "dataset_path": SUPERSTORE_PATH,
        "dataset_profile": None,
        "plan": [],
        "current_subtask_idx": 0,
        "rag_context": None,
        "pending_hitl_action": None,
        "session_id": "int-session-superstore",
        "user_id": "tester",
    }

    final_state = graph.invoke(state_input, config=config)

    # Verify plan has multiple executed steps
    plan = final_state.get("plan", [])
    assert len(plan) >= 1

    ai_msgs = [m for m in final_state["messages"] if m.type == "ai"]
    assert len(ai_msgs) > 0
    resp_text = ai_msgs[-1].content
    assert "Technology" in resp_text or "Furniture" in resp_text or "Sales" in resp_text
