# InsightForge — Campus Placement Resume & Technical Interview Guide

**Candidate Profile:** Achintya Singh (Akky)  
**Education:** 5th Year Undergraduate (Dual Degree), Indian Institute of Technology (IIT) Kharagpur  
**Target Profile:** Data & AI / Generative AI Engineer / Applied ML Scientist (Dec 2026 Placements)  
**Project Repository:** [github.com/Akky7684/insightforge](https://github.com/Akky7684/insightforge)

---

## 📄 1. Impact-Driven Resume Bullet Templates

Use these STAR-formatted, quantitative bullets directly on your IIT Kharagpur placement resume under **Projects / AI Engineering**:

### Version A (Comprehensive 4-Bullet Block — Recommended for Data & AI Shortlists):
- **InsightForge — Autonomous Multi-Agent Data Analyst** *(Python, LangGraph, FastAPI, ChromaDB, PostgreSQL, DuckDB, Docker)*
  - Engineered an autonomous 8-agent analytical state machine using **LangGraph**, decomposing natural-language queries into sandboxed AST-verified Python execution plans with a bounded self-correction Critic loop ($\le 2$ retries).
  - Developed a deterministic 91-question ground-truth evaluation benchmark across 5 diverse datasets (540k+ rows), empirically validating a **+38.9% accuracy lift** (98.9% vs 60.0%) over single-agent baseline LLMs at zero cloud API cost.
  - Architected zero-cost domain RAG grounding using **ChromaDB** ONNX embeddings for business metrics (AOV, Churn, Strike Rate), coupled with an embedded **DuckDB OLAP engine** executing sub-50ms columnar aggregations on million-row CSVs.
  - Implemented 4-trigger Human-in-the-Loop (HITL) safety guardrails (blocking destructive code, compute intensity, and PII leaks) with immutable **PostgreSQL audit logging**, containerized in a 4-tier **Docker Compose** stack achieving **3.51 RPS** under Locust load stress.

### Version B (Concise 3-Bullet Block for High-Density Resumes):
- **InsightForge — Multi-Agent Data Analytics & Forensic Engine** *(LangGraph, Gemini, DuckDB, ChromaDB, PostgreSQL)*
  - Designed an autonomous 8-agent analytical pipeline featuring AST-hardened sandboxed execution and a bounded self-correcting statistical Critic, achieving **98.9% accuracy** on a 91-question benchmark (+38.9% lift over direct LLMs).
  - Built Auto-ML predictive modeling and anomaly detection suites (Isolation Forest, LOF, Z-Score), embedded **DuckDB** for zero-copy OLAP SQL queries, and integrated in-memory caching to slash profiling latency by 95% (48ms P50).
  - Containerized full-stack deployment (FastAPI, Streamlit, PostgreSQL 16, ChromaDB) with 4-tier HITL safety guardrails and audit logging, maintaining 100% reliability at 3.51 RPS in concurrent Locust load testing.

---

## 🎙️ 2. Master Technical Interview Prep (Answering Section 22 Questions)

Here are the master architectural explanations to rehearse out loud for your live technical placement interviews:

### Q1: *"Why did you use LangGraph StateGraph instead of simpler frameworks like CrewAI, AutoGen, or standard LangChain chains?"*
> **Answer:**  
> "Standard sequential chains fail in production data analysis because analytics is inherently non-linear and exploratory. If a generated Python script fails a syntax check or statistical assumption (like normality for an ANOVA test), a linear chain crashes or produces hallucinations.  
> We chose **LangGraph** because it models the workflow as a stateful cyclic graph (`StateGraph`). This gave us:  
> 1. **Bounded Cyclic Self-Correction**: When the `Critic` detects an error, it routes state back to the `Coder` with specific error context, while enforcing a hard bound ($\le 2$ retries) to eliminate non-terminating loops.  
> 2. **Explicit State Schema**: Every node reads from and writes to a strongly-typed `InsightForgeState` containing verified subtasks, dataset profiles, and domain glossary context.  
> 3. **Production Checkpointing**: Seamless state persistence via `PostgresSaver`, enabling Human-in-the-Loop pause/resume without losing conversation context."

---

### Q2: *"How did you prevent the multi-agent system from getting stuck in an infinite loop when the Coder keeps writing faulty code?"*
> **Answer:**  
> "We implemented a 3-layer safeguard:  
> 1. **State-Tracked Retry Counter**: The `InsightForgeState` maintains an explicit `current_subtask_idx` and subtask retry tally.  
> 2. **Hard Upper Bound ($\le 2$ Retries)**: In `critic.py`, if a subtask fails more than twice, the Critic halts the retry loop, records the partial failure in the execution trace, and transitions state directly to the `Reporter` agent.  
> 3. **Fallback Narrative Generation**: The `Reporter` synthesizes what was successfully computed, explains why the specific subtask failed, and provides transparent guidance rather than hallucinating an answer."

---

### Q3: *"LLM-generated code execution can be extremely dangerous. How did you secure your Python execution sandbox?"*
> **Answer:**  
> "We used multi-layered defense-in-depth:  
> 1. **AST Static Analysis (`ast.parse`)**: Before execution, our sandbox inspects the Abstract Syntax Tree to enforce a strict whitelist of safe modules (`pandas`, `numpy`, `scipy`, `sklearn`, `matplotlib`, `seaborn`, `math`). It blocks dangerous AST nodes like `Import` or `ImportFrom` of `os`, `sys`, `subprocess`, `shutil`, `socket`, and `builtins.__import__`.  
> 2. **Restricted Global Namespace**: The execution environment overrides `__builtins__` to remove `open()`, `eval()`, `exec()`, `compile()`, and `globals()`.  
> 3. **Process Timeout & Memory Bounds**: Subprocess execution is capped at 15 seconds to prevent runaway CPU loops or memory denial-of-service."

---

### Q4: *"How did you prove that the multi-agent architecture is actually better than just asking a modern LLM (like Gemini or GPT-4) directly?"*
> **Answer:**  
> "We conducted an empirical ablation study across all 5 benchmark datasets comparing:  
> - **Config A (Single-Agent Baseline)**: Direct LLM zero-shot response without code execution or Critic validation.  
> - **Config B (InsightForge Multi-Agent)**: The full 6-agent sandboxed state machine with domain RAG grounding and Critic feedback.  
>  
> The single-agent baseline scored only **60.0% accuracy** because LLMs fundamentally hallucinate exact numerical aggregations on large datasets (e.g. guessing 10,981 sixes in IPL when the real count is 4,927). In contrast, InsightForge scored **98.9% accuracy** (a **+38.9% empirical accuracy lift**) because the Coder computes numbers deterministically in Python and the Critic verifies the result against schema bounds."

---

### Q5: *"Why did you integrate DuckDB alongside pandas?"*
> **Answer:**  
> "Pandas is great for expressive feature engineering, but it loads the entire dataset into uncompressed Python memory (creating a $5\text{x}-10\text{x}$ memory footprint) and executes single-threaded Python operations.  
> For analytical aggregations (group-bys, sums, counts) on large datasets (like our 540k-row e-commerce dataset), **DuckDB** provides:  
> 1. **Zero-Copy Vectorized Execution**: It queries the CSV/Parquet file directly using columnar SIMD vectorization without loading full tables into Python RAM.  
> 2. **Sub-50ms Latencies**: Complex multi-column aggregations complete in $< 30\text{ms}$.  
> 3. **Familiar SQL Interface**: Users can run instant read-only SQL queries directly from our UI console."

---

### Q6: *"How does your domain RAG grounding work, and why not just inject everything into the prompt?"*
> **Answer:**  
> "In enterprise business analytics, metric definitions are often ambiguous (e.g. *'How is Average Order Value (AOV) calculated?'* or *'What constitutes Net Churn?'*).  
> Stuffing every possible business formula into the prompt clutters the LLM context window, increases token latency, and causes attention degradation.  
> We built a local **ChromaDB vector store** using ONNX local embeddings (`all-MiniLM-L6-v2`). Before planning, the `RAG Agent` takes the user's query, performs cosine similarity retrieval over registered business formulas, and injects only the relevant mathematical and pandas formulas into the Planner and Coder context."

---

### Q7: *"What are the 4 Human-in-the-Loop (HITL) triggers you implemented, and how do they work in production?"*
> **Answer:**  
> "Our HITL guardrail layer evaluates code and queries before execution against 4 risk categories:  
> 1. `CODE_DESTRUCTIVE`: Flags state-mutating commands (e.g. `inplace=True`, file drops/deletions, SQL drops).  
> 2. `COMPUTE_INTENSIVE`: Flags high-complexity operations (Cartesian products `merge(how='cross')`, deep nested loops on $> 500\text{k}$ cells).  
> 3. `COUNTER_INTUITIVE`: Flags causal inference claims on observational data (e.g. Simpson's paradox assertions) requiring domain expert sign-off.  
> 4. `PII_SENSITIVE`: Scans for regex patterns matching Social Security Numbers (SSN), credit cards, emails, or plaintext credentials.  
>  
> When triggered, LangGraph sets `pending_hitl_action` and pauses. The user is presented with an interactive approval card in the UI, and the final decision is immutably logged into PostgreSQL `audit_logs` for regulatory compliance."
