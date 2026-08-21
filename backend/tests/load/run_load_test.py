"""Automated Headless Load Test Runner.

Executes concurrent simulated user requests against FastAPI application
and outputs a quantitative latency & throughput performance scorecard.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app

DATA_DIR = PROJECT_ROOT / "data"
TITANIC_PATH = str((DATA_DIR / "titanic.csv").resolve())
SUPERSTORE_PATH = str((DATA_DIR / "superstore.csv").resolve())


def make_request(client: TestClient, task_type: str) -> dict:
    """Execute a single simulated API request and record response time."""
    t0 = time.time()
    status = 500
    error = None

    try:
        if task_type == "health":
            resp = client.get("/health")
            status = resp.status_code
        elif task_type == "sample_datasets":
            resp = client.get("/api/sample-datasets")
            status = resp.status_code
        elif task_type == "profile_titanic":
            resp = client.get(f"/api/profile?dataset_path={TITANIC_PATH}")
            status = resp.status_code
        elif task_type == "profile_superstore":
            resp = client.get(f"/api/profile?dataset_path={SUPERSTORE_PATH}")
            status = resp.status_code
        elif task_type == "eda_titanic":
            resp = client.post(f"/api/eda/generate?dataset_path={TITANIC_PATH}")
            status = resp.status_code
        elif task_type == "predictive_titanic":
            resp = client.post(f"/api/predictive/train?dataset_path={TITANIC_PATH}&target_column=Survived")
            status = resp.status_code
        else:
            resp = client.get("/health")
            status = resp.status_code
    except Exception as e:
        error = str(e)

    lat_ms = (time.time() - t0) * 1000.0
    return {
        "task_type": task_type,
        "status": status,
        "latency_ms": lat_ms,
        "success": status == 200 and error is None,
        "error": error,
    }


def run_concurrency_benchmark(num_users: int = 10, requests_per_user: int = 5):
    """Run concurrent load test simulating multiple analyst users."""
    print("=" * 80)
    print(f"INSIGHTFORGE CONCURRENT LOAD TEST BENCHMARK ({num_users} Concurrent Users)")
    print("=" * 80)

    client = TestClient(app)
    # Warm up caches
    client.get("/health")
    client.get(f"/api/profile?dataset_path={TITANIC_PATH}")

    tasks = [
        "health",
        "sample_datasets",
        "profile_titanic",
        "profile_titanic", # Tests in-memory cache hit
        "profile_superstore",
        "eda_titanic",
        "predictive_titanic",
    ]

    total_requests = num_users * requests_per_user
    task_queue = [tasks[i % len(tasks)] for i in range(total_requests)]

    results = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [executor.submit(make_request, client, t) for t in task_queue]
        for f in as_completed(futures):
            results.append(f.result())

    total_time = time.time() - t_start
    rps = len(results) / total_time if total_time > 0 else 0.0

    # Calculate metrics
    latencies = [r["latency_ms"] for r in results]
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    fail_rate = (fail_count / len(results)) * 100.0

    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    avg_lat = float(np.mean(latencies))

    print(f"Total Requests Executed:    {len(results)}")
    print(f"Total Test Duration:        {total_time:.2f}s")
    print(f"Throughput (RPS):           {rps:.2f} req/s")
    print(f"Success Rate:               {(success_count / len(results)) * 100:.1f}%")
    print(f"Failure Rate:               {fail_rate:.1f}%")
    print(f"Latency P50 (Median):       {p50:.2f} ms")
    print(f"Latency P90:                {p90:.2f} ms")
    print(f"Latency P95:                {p95:.2f} ms")
    print(f"Latency P99:                {p99:.2f} ms")
    print(f"Average Latency:            {avg_lat:.2f} ms")
    print("=" * 80)

    # Per-Endpoint Breakdown
    df_res = pd.DataFrame(results)
    summary_df = df_res.groupby("task_type").agg(
        count=("latency_ms", "count"),
        p50_ms=("latency_ms", lambda x: np.percentile(x, 50)),
        p95_ms=("latency_ms", lambda x: np.percentile(x, 95)),
        avg_ms=("latency_ms", "mean"),
    ).reset_index()

    print("\nPER-ENDPOINT LATENCY BREAKDOWN:")
    print(summary_df.to_string(index=False))
    print("=" * 80)

    return {
        "total_requests": len(results),
        "rps": round(rps, 2),
        "success_rate_pct": round((success_count / len(results)) * 100, 2),
        "failure_rate_pct": round(fail_rate, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p90_ms": round(p90, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "avg_latency_ms": round(avg_lat, 2),
    }


if __name__ == "__main__":
    run_concurrency_benchmark(num_users=8, requests_per_user=4)
