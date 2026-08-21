# InsightForge — Quantitative Evaluation Results

**Last Benchmark Run:** August 21, 2026  
**Evaluation Scope:** 91 Ground-Truth Analytical Questions across all 5 Production Datasets

---

## 1. Datasets Under Evaluation
1. **Titanic Passenger Survival** (`titanic.csv` — 891 rows, 12 columns)
2. **Retail Superstore Sales & Profit** (`superstore.csv` — 9,994 rows, 21 columns)
3. **UCI E-Commerce Transactions** (`ecommerce.csv` — 541,909 rows, 8 columns)
4. **IPL Ball-by-Ball Cricket** (`ipl.csv` — 99,120 deliveries, 18 columns)
5. **Synthetic Transactions with Injected Outliers** (`synthetic_anomaly.csv` — 1,000 rows, 50 labeled anomalies)

---

## 2. Final Summary Metric Scorecard (91-Question Comprehensive Benchmark Lock)

| Metric | Target | Final Actual Result (91 Qs) | Status |
|---|---|---|---|
| **Task Success Rate** | $\ge 95\%$ | **100.0%** (91/91 tasks completed) | 🏆 Exceeds Target |
| **Answer Accuracy** | $\ge 88\%$ | **98.9%** (90/91 questions verified correct) | 🏆 Production Ready |
| **Self-Correction Recovery Rate** | $\ge 80\%$ | **100.0%** (All retries resolved $\le 2$ iterations) | 🏆 100% Resolved |
| **Avg. Iterations to Success** | $\le 1.3$ | **1.18 iterations** | ⚡ High Precision |
| **Single-Agent vs Multi-Agent Accuracy Lift** | $\ge +30\%$ | **+38.9% Lift** (98.9% vs 60.0%) | 🚀 Huge Empirical Lift |
| **Latency P50 (Median)** | $< 8\text{s}$ | **5.69s** | ⚡ Sub-6-Second |
| **Latency P90** | $< 20\text{s}$ | **6.76s** | ⚡ Rapid Response |
| **Average Latency** | $< 12\text{s}$ | **5.77s** | ⚡ High Throughput |
| **Cost per Query** | $< \$0.005$ | **\$0.00** (Gemini Free Tier) | 💰 Zero Cost |

---

## 3. Comparative Ablation Study: Single-Agent vs Multi-Agent Pipeline

To empirically validate the multi-agent design, we executed a comparative ablation study across all 5 benchmark datasets:

| Configuration | Architecture | Accuracy | Failure Mode / Hallucinations | Avg Latency |
|---|---|---|---|---|
| **Config A: Single-Agent Baseline** | Direct LLM (no sandbox / no Critic / no RAG) | **60.0%** | Hallucinates dataset totals, sums, and exact metrics (e.g. 10,981 vs 4,927 sixes) | **1.31s** |
| **Config B: InsightForge Multi-Agent** | 6-Agent LangGraph (AST Sandbox + RAG + Critic) | **98.9%** | None (Code verified and executed in AST sandbox) | **5.77s** |

> **Key Takeaway:** Sandboxed AST Python code execution combined with domain RAG formula grounding and Critic self-correction reflection provides a **+38.9% empirical accuracy lift** over a naive LLM baseline while maintaining sub-6-second average latency at zero cost.

---

## 4. Per-Dataset Breakdown (91 Questions)

| Dataset | Total Questions | Task Success Rate | Accuracy | Key Competencies Verified |
|---|---|---|---|---|
| **`titanic.csv`** | 25 | 100.0% | 100.0% | Distributions, grouping, multi-step odds ratios, hypothesis testing |
| **`superstore.csv`** | 20 | 100.0% | 100.0% | Multi-step subcategory ranking, profit margins, regional discounts |
| **`ecommerce.csv`** | 15 | 100.0% | 100.0% | High-scale 541k aggregation, AOV formula grounding, invoice cancellations |
| **`ipl.csv`** | 15 | 100.0% | 100.0% | 99k ball-by-ball sports analytics, leading batters, boundaries, extras |
| **`synthetic_anomaly.csv`** | 16 | 100.0% | 93.8% | Outlier detection, transaction volume, max amount, regional breakdowns |

---

## 5. Concurrency & Load Stress Test Scorecard (Locust Benchmark — Week 13)

| Metric | Target | Actual Result | Status |
|---|---|---|---|
| **Simulated Concurrent Users** | 5–10 concurrent users | **8 Users** (Multi-persona workload) | ✅ Validated |
| **Request Success Rate** | $\ge 99.0\%$ | **100.0%** (0.0% failure rate) | 🏆 Perfect Reliability |
| **Throughput (RPS)** | $\ge 2.0\text{ req/s}$ | **3.51 req/s** | ✅ Exceeds Target |
| **Overall Latency P50 (Median)** | $< 500\text{ms}$ | **208.64 ms** | ⚡ High Speed |
| **Cached Profile Query P50** | $< 100\text{ms}$ | **48.76 ms** (95% reduction via in-memory cache) | 🚀 Sub-50ms |
| **API Health Ping P50** | $< 50\text{ms}$ | **34.79 ms** | ⚡ Instant |

---

## 6. Evaluation Methodology
- **Benchmark Runner**: [`backend/tests/eval/run_eval.py`](../backend/tests/eval/run_eval.py)
- **Ablation Study**: [`backend/tests/eval/run_ablation.py`](../backend/tests/eval/run_ablation.py)
- **Load Test Harness**: [`backend/tests/load/locustfile.py`](../backend/tests/load/locustfile.py) & [`backend/tests/load/run_load_test.py`](../backend/tests/load/run_load_test.py)
- **Benchmark Source**: [`backend/tests/eval/benchmark.json`](../backend/tests/eval/benchmark.json)
- **Ground Truth Establishment**: Computed deterministically via offline pandas scripts and cross-verified against dataset schema.
