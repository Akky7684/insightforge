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

## 2. Summary Metric Scorecard

| Metric | Target | Week 3 Baseline (10 Qs) | Final Target (90 Qs) |
|---|---|---|---|
| **Task Success Rate** | $\ge 95\%$ | **100.0%** (10/10) | $\ge 95\%$ |
| **Answer Accuracy** | $\ge 85\%$ | **100.0%** (10/10) | $\ge 88\%$ |
| **Latency P50** | $< 10\text{s}$ | **6.93s** | $< 8\text{s}$ |
| **Latency P90** | $< 25\text{s}$ | **22.94s** | $< 20\text{s}$ |
| **Average Latency** | $< 15\text{s}$ | **11.82s** | $< 12\text{s}$ |
| **Cost per Query** | $< \$0.02$ | **\$0.00** (Gemini Free Tier) | $< \$0.005$ |

---

## 3. Question-by-Question Breakdown

| ID | Dataset | Question | Expected Ground Truth | Agent Output | Result | Latency |
|---|---|---|---|---|---|---|
| `eval_01` | `titanic.csv` | Total passengers count | `891` | `"Total number of passengers: 891"` | **PASS** | 7.62s |
| `eval_02` | `titanic.csv` | Overall survival rate % | `38.38%` | `"Overall survival rate: 38.38%"` | **PASS** | 21.77s |
| `eval_03` | `titanic.csv` | Average passenger age | `29.70` | `"Average age of passengers: 29.70"` | **PASS** | 4.85s |
| `eval_04` | `titanic.csv` | Gender breakdown count | `male: 577, female: 314` | `"Male: 577, Female: 314"` | **PASS** | 19.69s |
| `eval_05` | `titanic.csv` | 1st class average fare | `$84.15` | `"Average fare paid: $84.15"` | **PASS** | 10.37s |
| `eval_06` | `superstore.csv` | Total sales sum | `$2,297,200.86` | `"Total Sales: $2,297,200.86"` | **PASS** | 5.09s |
| `eval_07` | `superstore.csv` | Total profit sum | `$286,397.02` | `"Total Profit: $286,397.02"` | **PASS** | 33.45s |
| `eval_08` | `superstore.csv` | Category with highest sales | `Technology` | `"Highest Total Sales: Technology ($836,154.03)"` | **PASS** | 6.24s |
| `eval_09` | `superstore.csv` | Unique order IDs count | `5009` | `"Number of unique orders: 5009"` | **PASS** | 4.55s |
| `eval_10` | `superstore.csv` | Total sales in West region | `$725,457.82` | `"Total sales for West: $725,457.82"` | **PASS** | 4.61s |

---

## 4. Evaluation Methodology
- **Harness Runner**: [`backend/tests/eval/run_eval.py`](../backend/tests/eval/run_eval.py)
- **Benchmark Source**: [`backend/tests/eval/benchmark.json`](../backend/tests/eval/benchmark.json)
- **Ground Truth Establishment**: Computed deterministically via offline pandas scripts and cross-verified against dataset schema.
