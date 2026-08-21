# InsightForge — Quantitative Evaluation Results

**Last Benchmark Run:** August 21, 2026  
**Evaluation Scope:** 10 Ground-Truth Analytical Questions across 2 Datasets (`titanic.csv`, `superstore.csv`)

---

## 1. Datasets Under Evaluation
1. **Titanic Passenger Survival** (`titanic.csv` — 891 rows, 12 columns)
2. **Retail Superstore Sales & Profit** (`superstore.csv` — 9,994 rows, 21 columns)
3. **UCI E-Commerce Transactions** (`ecommerce.csv` — 541,909 rows, 8 columns)
4. **IPL Ball-by-Ball Cricket** (`ipl.csv` — 99,120 deliveries, 18 columns)
5. **Synthetic Transactions with Injected Outliers** (`synthetic_anomaly.csv` — 1,000 rows, 50 labeled anomalies)

---

## 2. Summary Metric Scorecard (Midterm Evaluation — Week 8 Lock)

| Metric | Target | Week 8 Midterm Actuals | Final Target (90 Qs) |
|---|---|---|---|
| **Task Success Rate** | $\ge 95\%$ | **100.0%** (35/35) | $\ge 95\%$ |
| **Answer Accuracy** | $\ge 85\%$ | **100.0%** (35/35) | $\ge 88\%$ |
| **Self-Correction Recovery Rate** | $\ge 75\%$ | **100.0%** (All retries resolved $\le 2$) | $\ge 80\%$ |
| **Avg. Iterations to Success** | $\le 1.5$ | **1.2 iterations** | $\le 1.3$ |
| **Single-Agent vs Multi-Agent Accuracy Lift** | $\ge +25\%$ | **+40.0% Lift** (100% vs 60%) | $\ge +30\%$ |
| **Latency P50** | $< 10\text{s}$ | **4.96s** | $< 8\text{s}$ |
| **Latency P90** | $< 25\text{s}$ | **7.00s** | $< 20\text{s}$ |
| **Average Latency** | $< 15\text{s}$ | **5.49s** | $< 12\text{s}$ |
| **Cost per Query** | $< \$0.02$ | **\$0.00** (Gemini Free Tier) | $< \$0.005$ |

---

## 3. Comparative Ablation Study: Single-Agent vs Multi-Agent Pipeline

To empirically validate the multi-agent design, we executed a comparative ablation study across all 5 benchmark datasets:

| Configuration | Architecture | Accuracy | Failure Mode / Hallucinations | Avg Latency |
|---|---|---|---|---|
| **Config A: Single-Agent Baseline** | Direct LLM (no sandbox / no Critic / no RAG) | **60.0%** | Hallucinates dataset totals, sums, and exact metrics (e.g. 10,981 vs 4,927 sixes) | **1.31s** |
| **Config B: InsightForge Multi-Agent** | 6-Agent LangGraph (AST Sandbox + RAG + Critic) | **100.0%** | None (Code verified and executed on actual data) | **5.49s** |

> **Key Takeaway:** Sandboxed AST Python code execution combined with domain RAG formula grounding and Critic self-correction reflection provides a **+40.0% empirical accuracy lift** over a naive LLM baseline while maintaining sub-6-second average latency at zero cost.

---

## 4. Per-Dataset Breakdown (35 Questions)

| Dataset | Total Questions | Task Success Rate | Accuracy | Key Competencies Verified |
|---|---|---|---|---|
| **`titanic.csv`** | 15 | 100.0% | 100.0% | Distributions, grouping, multi-step odds ratios, hypothesis testing |
| **`superstore.csv`** | 5 | 100.0% | 100.0% | Multi-step subcategory ranking, profit margins, regional discounts |
| **`ecommerce.csv`** | 5 | 100.0% | 100.0% | Revenue aggregation, AOV formula grounding, invoice cancellations |
| **`ipl.csv`** | 5 | 100.0% | 100.0% | Ball-by-ball delivery counts, leading run scorers, boundary counts |
| **`synthetic_anomaly.csv`** | 5 | 100.0% | 100.0% | Transaction volume, max amount, regional breakdown, customer age |

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
