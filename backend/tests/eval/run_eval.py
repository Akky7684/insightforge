"""InsightForge — Evaluation Benchmark Runner.

Executes ground-truth benchmark questions across evaluation datasets,
measures accuracy, success rate, and latency, and formats quantitative metrics.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Setup paths & sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BENCHMARK_PATH = PROJECT_ROOT / "backend" / "tests" / "eval" / "benchmark.json"
DATA_DIR = PROJECT_ROOT / "data"

import numpy as np
from langchain_core.messages import HumanMessage

from backend.app.config import get_settings
from backend.app.graph.supervisor import get_graph


def check_answer_match(response_text: str, item: Dict[str, Any]) -> bool:
    """Evaluate whether the agent response contains the ground truth value within tolerance."""
    text = response_text.replace(",", "").replace("$", "").lower()
    
    # 1. Direct acceptable strings match
    for acc in item.get("acceptable_strings", []):
        cleaned_acc = acc.replace(",", "").replace("$", "").lower()
        if cleaned_acc in text:
            return True

    # 2. Numeric tolerance check
    gt = item.get("ground_truth")
    if isinstance(gt, (int, float)):
        # Extract all floats/ints from the response text
        numbers_in_text = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        for num_str in numbers_in_text:
            try:
                num_val = float(num_str)
                if abs(num_val - gt) <= max(0.01 * gt * (item.get("tolerance_pct", 1.0) / 100.0), 0.05):
                    return True
            except ValueError:
                continue

    # 3. Dict matching (e.g. {"male": 577, "female": 314})
    elif isinstance(gt, dict):
        matches = 0
        for k, v in gt.items():
            if str(v) in text:
                matches += 1
        if matches == len(gt):
            return True

    # 4. String matching
    elif isinstance(gt, str):
        if gt.lower() in text:
            return True

    return False


def run_evaluation(benchmark_file: Path = BENCHMARK_PATH) -> Dict[str, Any]:
    """Execute evaluation benchmark and return structured metrics."""
    with open(benchmark_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    print("=" * 70, flush=True)
    print(f"INSIGHTFORGE BENCHMARK EVALUATION ({len(questions)} Questions)", flush=True)
    print("=" * 70, flush=True)

    graph = get_graph()
    results = []
    latencies = []

    for i, item in enumerate(questions, 1):
        q_id = item["id"]
        dataset_name = item["dataset"]
        dataset_path = str((DATA_DIR / dataset_name).resolve())
        query = item["question"]

        print(f"\n[{i}/{len(questions)}] ({q_id}) [{dataset_name}]", flush=True)
        print(f"  Question: {query}", flush=True)

        t0 = time.time()
        error_msg = None
        response_text = ""
        passed = False

        state_input = {
            "messages": [HumanMessage(content=query)],
            "dataset_id": dataset_name,
            "dataset_path": dataset_path,
            "dataset_profile": None,
            "plan": [],
            "current_subtask_idx": 0,
            "rag_context": None,
            "pending_hitl_action": None,
            "session_id": f"eval_{q_id}_{int(time.time())}",
            "user_id": "eval_runner",
        }

        try:
            config = {"configurable": {"thread_id": f"eval_thread_{q_id}_{time.time()}"}}
            res = graph.invoke(state_input, config=config)
            ai_msgs = [m for m in res["messages"] if m.type == "ai"]
            response_text = ai_msgs[-1].content.strip() if ai_msgs else "No response"
            passed = check_answer_match(response_text, item)
            
            latency = time.time() - t0
            latencies.append(latency)

            status_str = "PASS" if passed else "FAIL"
            clean_resp_snippet = response_text.replace("\n", " ")[:90].encode("ascii", "replace").decode("ascii")
            print(f"  Response: {clean_resp_snippet}...", flush=True)
            print(f"  Result: [{status_str}] in {latency:.2f}s", flush=True)
        except Exception as e:
            error_msg = str(e)
            response_text = f"CRASH: {e}"
            # Check accuracy
            passed = check_answer_match(response_text, item)
            lat = time.time() - t0

            status_str = "[PASS]" if passed else "[FAIL]"
            clean_resp_snippet = response_text.replace("\n", " ")[:90].encode("ascii", "replace").decode("ascii")
            print(f"  Response: {clean_resp_snippet}...", flush=True)
            print(f"  Result: {status_str} in {lat:.2f}s", flush=True)

        results.append({
            "id": q_id,
            "dataset": dataset_name,
            "question": query,
            "expected": item.get("ground_truth"),
            "response": response_text,
            "passed": passed,
            "crashed": error_msg is not None,
            "latency": round(latency, 2),
        })

        # Pacing delay between questions to respect free-tier rate limits
        time.sleep(3)

    # Metric computations
    total_q = len(results)
    successful_runs = sum(1 for r in results if not r["crashed"])
    correct_answers = sum(1 for r in results if r["passed"])

    task_success_rate = round((successful_runs / total_q) * 100, 2) if total_q > 0 else 0.0
    accuracy_rate = round((correct_answers / total_q) * 100, 2) if total_q > 0 else 0.0
    p50_latency = round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0
    p90_latency = round(float(np.percentile(latencies, 90)), 2) if latencies else 0.0
    avg_latency = round(float(np.mean(latencies)), 2) if latencies else 0.0

    print("\n" + "=" * 70, flush=True)
    print("EVALUATION SUMMARY SCORECARD", flush=True)
    print("=" * 70, flush=True)
    print(f"Total Questions Evaluated:    {total_q}", flush=True)
    print(f"Task Success Rate:            {task_success_rate}% ({successful_runs}/{total_q})", flush=True)
    print(f"Answer Accuracy:              {accuracy_rate}% ({correct_answers}/{total_q})", flush=True)
    print(f"Latency (Avg / P50 / P90):    {avg_latency}s / {p50_latency}s / {p90_latency}s", flush=True)
    print("=" * 70, flush=True)

    return {
        "total_questions": total_q,
        "task_success_rate": task_success_rate,
        "answer_accuracy": accuracy_rate,
        "avg_latency": avg_latency,
        "p50_latency": p50_latency,
        "p90_latency": p90_latency,
        "results": results,
    }


if __name__ == "__main__":
    run_evaluation()
