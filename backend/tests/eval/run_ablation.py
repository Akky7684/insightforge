"""Comparative Ablation Study — Single-Agent Baseline vs InsightForge Multi-Agent Pipeline.

Evaluates 10 analytical questions across Titanic, Superstore, E-Commerce, IPL Cricket,
and Synthetic Anomaly datasets comparing:
- Configuration A: Baseline Single-Agent (Direct LLM without sandbox execution/Critic/RAG)
- Configuration B: InsightForge Multi-Agent Pipeline (Full 6-agent collaborative StateGraph)
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage

from backend.app.config import get_llm, get_settings
from backend.app.graph.supervisor import get_graph

DATA_DIR = Path(get_settings().data_dir)
BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark.json"


def evaluate_single_agent(question: str, dataset_name: str, schema_str: str) -> str:
    """Baseline Single-Agent: Direct LLM generation without execution sandbox or RAG."""
    llm = get_llm("flash")
    prompt = (
        f"You are a data assistant. Answer the following question about the dataset '{dataset_name}'.\n"
        f"Dataset columns: {schema_str}\n\n"
        f"Question: {question}\n\n"
        f"Provide the exact numerical answer or category directly based on your knowledge."
    )
    try:
        resp = llm.invoke(prompt)
        content = resp.content
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
            return " ".join(text_parts).strip()
        return str(content).strip()
    except Exception as e:
        return f"ERROR: {e}"


def evaluate_multi_agent(question: str, dataset_name: str, dataset_path: str) -> str:
    """InsightForge Multi-Agent Pipeline: Full StateGraph with sandbox execution, Critic, and RAG."""
    graph = get_graph()
    config = {"configurable": {"thread_id": f"ablation-{time.time()}"}}
    state_input = {
        "messages": [HumanMessage(content=question)],
        "dataset_id": dataset_name,
        "dataset_path": dataset_path,
        "dataset_profile": None,
        "plan": [],
        "current_subtask_idx": 0,
        "rag_context": None,
        "pending_hitl_action": None,
        "session_id": f"ablation-{dataset_name}",
        "user_id": "evaluator",
    }
    try:
        final_state = graph.invoke(state_input, config=config)
        ai_msgs = [m for m in final_state["messages"] if m.type == "ai"]
        return ai_msgs[-1].content if ai_msgs else "No response"
    except Exception as e:
        return f"CRASH: {e}"


def check_answer_pass(response_text: str, item: dict) -> bool:
    """Validate if answer contains ground-truth target or acceptable strings."""
    acceptable = item.get("acceptable_strings", [])
    res_lower = response_text.lower()
    for acc in acceptable:
        if acc.lower() in res_lower:
            return True
    return False


def run_ablation():
    """Execute ablation benchmark comparing Single-Agent vs Multi-Agent."""
    print("=" * 80)
    print("INSIGHTFORGE ABLATION STUDY: SINGLE-AGENT BASELINE VS MULTI-AGENT PIPELINE")
    print("=" * 80)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    # Select 10 representative questions (2 per dataset)
    selected_indices = [0, 4, 5, 9, 20, 23, 25, 27, 30, 32]
    questions = [bench_data["questions"][i] for i in selected_indices if i < len(bench_data["questions"])]

    single_results = []
    multi_results = []

    for i, q in enumerate(questions, 1):
        q_text = q["question"]
        ds_name = q["dataset"]
        ds_path = str((DATA_DIR / ds_name).resolve())

        print(f"\n[{i}/{len(questions)}] ({q['id']}) [{ds_name}]")
        print(f"  Question: {q_text}")
        print(f"  Ground Truth Target: {q['ground_truth']}")

        # 1. Evaluate Single-Agent Baseline
        t0 = time.time()
        single_resp = evaluate_single_agent(q_text, ds_name, "Columns present in CSV")
        single_lat = time.time() - t0
        single_pass = check_answer_pass(single_resp, q)
        single_results.append({"pass": single_pass, "latency": single_lat, "resp": single_resp})
        print(f"  Single-Agent Baseline: [{'PASS' if single_pass else 'FAIL'}] ({single_lat:.2f}s) -> {single_resp[:80]}...")

        # Small pacing delay
        time.sleep(2)

        # 2. Evaluate Multi-Agent Architecture
        t1 = time.time()
        multi_resp = evaluate_multi_agent(q_text, ds_name, ds_path)
        multi_lat = time.time() - t1
        multi_pass = check_answer_pass(multi_resp, q)
        multi_results.append({"pass": multi_pass, "latency": multi_lat, "resp": multi_resp})
        print(f"  InsightForge Multi-Agent: [{'PASS' if multi_pass else 'FAIL'}] ({multi_lat:.2f}s) -> {multi_resp[:80]}...")

        time.sleep(3)

    # Calculate aggregate scores
    single_acc = (sum(1 for r in single_results if r["pass"]) / len(single_results)) * 100
    multi_acc = (sum(1 for r in multi_results if r["pass"]) / len(multi_results)) * 100
    single_avg_lat = sum(r["latency"] for r in single_results) / len(single_results)
    multi_avg_lat = sum(r["latency"] for r in multi_results) / len(multi_results)

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPARATIVE RESULTS SCORECARD")
    print("=" * 80)
    print(f"Total Test Cases:                {len(questions)}")
    print(f"Single-Agent Baseline Accuracy:  {single_acc:.1f}%")
    print(f"InsightForge Multi-Agent Acc:    {multi_acc:.1f}%")
    print(f"Empirical Accuracy Improvement:  +{multi_acc - single_acc:.1f}% (Lift)")
    print(f"Single-Agent Avg Latency:        {single_avg_lat:.2f}s")
    print(f"Multi-Agent Avg Latency:         {multi_avg_lat:.2f}s")
    print("=" * 80)

    # Save ablation summary
    ablation_out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "single_agent_accuracy": single_acc,
        "multi_agent_accuracy": multi_acc,
        "accuracy_lift_pct": multi_acc - single_acc,
        "single_agent_avg_latency": round(single_avg_lat, 2),
        "multi_agent_avg_latency": round(multi_avg_lat, 2),
    }
    with open(Path(__file__).resolve().parent / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_out, f, indent=2)


if __name__ == "__main__":
    run_ablation()
