"""InsightForge — Streamlit Frontend UI.

Interactive multi-agent data analyst interface supporting CSV/Excel uploads,
automated dataset profiling, chart visualization, and natural-language Q&A.
"""

import os
import re
import sys
import uuid
from pathlib import Path

# Ensure project root is in sys.path so 'backend' package is always resolvable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd
import requests
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="InsightForge | Autonomous Multi-Agent Data Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Backend API URL ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: -webkit-linear-gradient(45deg, #2b5876, #4e4376);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #666;
        margin-bottom: 1.2rem;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #4e4376;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_dataset_path" not in st.session_state:
    st.session_state.active_dataset_path = None

if "active_dataset_name" not in st.session_state:
    st.session_state.active_dataset_name = None

if "dataset_df" not in st.session_state:
    st.session_state.dataset_df = None

if "dataset_profile" not in st.session_state:
    st.session_state.dataset_profile = None


# --- Helper Functions ---
def load_dataset(file_path: str, file_name: str):
    """Load and preview dataset in session state, generating profile."""
    try:
        df = pd.read_csv(file_path, encoding="latin1")
        st.session_state.dataset_df = df
        st.session_state.active_dataset_path = str(Path(file_path).resolve())
        st.session_state.active_dataset_name = file_name

        # Trigger profiling
        from backend.app.graph.agents.profiler import profile_dataset
        st.session_state.dataset_profile = profile_dataset(file_path)
        return True
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")
        return False


def query_agent(prompt: str) -> dict:
    """Send natural language query to LangGraph multi-agent pipeline."""
    dataset_path = st.session_state.active_dataset_path
    if not dataset_path:
        return {"response": "⚠️ Please select or upload a dataset first!", "plan": []}

    try:
        from langchain_core.messages import HumanMessage
        from backend.app.graph.supervisor import get_graph

        graph = get_graph()
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        state_input = {
            "messages": [HumanMessage(content=prompt)],
            "dataset_id": st.session_state.active_dataset_name,
            "dataset_path": dataset_path,
            "dataset_profile": st.session_state.dataset_profile,
            "plan": [],
            "current_subtask_idx": 0,
            "rag_context": None,
            "pending_hitl_action": None,
            "session_id": st.session_state.session_id,
            "user_id": "user_1",
        }
        res = graph.invoke(state_input, config=config)
        ai_msgs = [m for m in res["messages"] if m.type == "ai"]
        ans_text = ai_msgs[-1].content if ai_msgs else "Analysis completed."
        plan_list = res.get("plan", [])
        return {"response": ans_text, "plan": plan_list}
    except Exception as e:
        return {"response": f"❌ Execution error: {e}", "plan": []}


def render_message_content(msg_data):
    """Render text message, subtask execution plan, and any embedded charts."""
    if isinstance(msg_data, dict):
        content = msg_data.get("content", "")
        plan = msg_data.get("plan", [])
    else:
        content = str(msg_data)
        plan = []

    # Display subtask execution plan expander if multi-step
    if plan and len(plan) > 1:
        with st.expander(f"🧩 Multi-Step Analytical Plan ({len(plan)} Steps Completed)"):
            for i, st_item in enumerate(plan, 1):
                st_desc = st_item.description if hasattr(st_item, "description") else st_item.get("description", "")
                st_res = st_item.result if hasattr(st_item, "result") else st_item.get("result", "")
                st.markdown(f"**Step {i}:** `{st_desc}`")
                if st_res:
                    # Clean out chart tags in subtask summary
                    clean_res = re.sub(r"\[CHART:.*?\]", "", st_res).strip()
                    st.caption(f"↳ {clean_res[:200]}..." if len(clean_res) > 200 else f"↳ {clean_res}")

    # Check for [CHART:<path>] tags
    chart_matches = re.findall(r"\[CHART:(.*?)\]", content)
    cleaned_text = re.sub(r"\[CHART:.*?\]", "", content).strip()

    if cleaned_text:
        st.markdown(cleaned_text)

    for chart_path in chart_matches:
        if os.path.exists(chart_path):
            st.image(chart_path, caption="Generated Visualization", use_container_width=True)
        else:
            st.caption(f"📊 Chart generated: `{Path(chart_path).name}`")

    # Display Multi-Agent Collaboration Badge
    if plan:
        st.caption("🤖 **Active Agents:** 🕵️ *Profiler* • 📚 *RAG Grounding* • 📋 *Planner* • 💻 *Coder* • 🛡️ *Critic* • 📝 *Reporter*")


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📊 Dataset Control Panel")

    data_source = st.radio("Choose source:", ["Sample Datasets", "Upload CSV / Excel"], index=0)

    if data_source == "Sample Datasets":
        sample_dir = Path(__file__).resolve().parent.parent / "data"
        sample_files = list(sample_dir.glob("*.csv"))

        if sample_files:
            file_options = {f.name: str(f.resolve()) for f in sample_files}
            selected_sample = st.selectbox(
                "Select benchmark dataset:",
                options=list(file_options.keys()),
                index=0 if "titanic.csv" not in file_options else list(file_options.keys()).index("titanic.csv"),
            )

            if st.button("Load Dataset", type="primary", use_container_width=True):
                if load_dataset(file_options[selected_sample], selected_sample):
                    st.success(f"Loaded: `{selected_sample}`")
                    st.session_state.messages = []
                    st.rerun()
        else:
            st.warning("No sample datasets found in `data/`.")

    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            if st.button("Process & Load File", type="primary", use_container_width=True):
                upload_dir = Path(__file__).resolve().parent.parent / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                save_path = upload_dir / f"upload_{uploaded_file.name}"

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if save_path.suffix.lower() in [".xlsx", ".xls"]:
                    df_temp = pd.read_excel(save_path)
                    csv_path = save_path.with_suffix(".csv")
                    df_temp.to_csv(csv_path, index=False)
                    save_path = csv_path

                if load_dataset(str(save_path), uploaded_file.name):
                    st.success(f"Uploaded `{uploaded_file.name}`")
                    st.session_state.messages = []
                    st.rerun()

    # Active dataset quick stats
    if st.session_state.dataset_df is not None:
        df = st.session_state.dataset_df
        st.markdown("---")
        st.markdown(f"**Active:** `{st.session_state.active_dataset_name}`")
        st.markdown(f"**Records:** `{df.shape[0]:,}` | **Features:** `{df.shape[1]}`")

        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()


# --- Main Dashboard ---
st.markdown('<div class="main-title">⚡ InsightForge</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous Multi-Agent Data Analyst — LangGraph Orchestration & Sandboxed Analytics</div>', unsafe_allow_html=True)

# Auto-load titanic.csv if session empty
if st.session_state.dataset_df is None:
    default_titanic = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"
    if default_titanic.exists():
        load_dataset(str(default_titanic), "titanic.csv")
        st.rerun()

tab_chat, tab_profile, tab_eda, tab_anomaly, tab_predictive, tab_glossary, tab_audit = st.tabs([
    "💬 Conversational Analyst",
    "📊 Dataset Deep Profile",
    "📑 Automated EDA & Insights",
    "🚨 Anomaly Detection",
    "🔮 Predictive Modeling",
    "📚 Business Glossary & RAG",
    "🛡️ Governance & Audit Logs"
])

# --- TAB 1: Chat Interface ---
with tab_chat:
    if st.session_state.active_dataset_name:
        st.caption(f"Active Context: `{st.session_state.active_dataset_name}` ({len(st.session_state.dataset_df):,} rows)")

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_message_content(msg)

    # Quick Action Buttons
    if not st.session_state.messages and st.session_state.active_dataset_name == "titanic.csv":
        st.markdown("**Suggested Analysis Prompts:**")
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Survival by Gender"):
            p = "What was the survival rate of male vs female passengers?"
            st.session_state.messages.append({"role": "user", "content": p})
            with st.spinner("Analyzing in sandbox..."):
                res = query_agent(p)
            st.session_state.messages.append({"role": "assistant", "content": res["response"], "plan": res["plan"]})
            st.rerun()

        if c2.button("Plot Class Survival"):
            p = "Plot a bar chart showing survival rate by passenger class (Pclass)."
            st.session_state.messages.append({"role": "user", "content": p})
            with st.spinner("Generating visualization..."):
                res = query_agent(p)
            st.session_state.messages.append({"role": "assistant", "content": res["response"], "plan": res["plan"]})
            st.rerun()

        if c3.button("Fare Distribution"):
            p = "What is the average, median, and max fare paid?"
            st.session_state.messages.append({"role": "user", "content": p})
            with st.spinner("Analyzing in sandbox..."):
                res = query_agent(p)
            st.session_state.messages.append({"role": "assistant", "content": res["response"], "plan": res["plan"]})
            st.rerun()

        if c4.button("Age Outliers"):
            p = "How many passengers have age values considered statistical outliers?"
            st.session_state.messages.append({"role": "user", "content": p})
            with st.spinner("Analyzing in sandbox..."):
                res = query_agent(p)
            st.session_state.messages.append({"role": "assistant", "content": res["response"], "plan": res["plan"]})
            st.rerun()

    # Chat Input
    if user_query := st.chat_input("Ask any question or request a plot (e.g. 'Plot sales by category', 'Calculate survival correlation')..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("InsightForge agents executing in sandbox..."):
                res = query_agent(user_query)
                render_message_content({"content": res["response"], "plan": res["plan"]})

        st.session_state.messages.append({"role": "assistant", "content": res["response"], "plan": res["plan"]})


# --- TAB 2: Dataset Deep Profile ---
with tab_profile:
    if st.session_state.dataset_profile:
        prof = st.session_state.dataset_profile
        st.markdown(f"### 📈 Automated Profile: `{prof['dataset_name']}`")

        # Metric KPI cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Rows", f"{prof['row_count']:,}")
        kpi2.metric("Total Columns", prof["column_count"])
        kpi3.metric("Missing Cells", f"{prof['total_missing_cells']:,}")
        kpi4.metric("High Correlations (|r|>=0.5)", len(prof["high_correlations"]))

        st.markdown("---")

        # Column Schema & Missingness Table
        st.markdown("#### 📋 Columns & Missingness")
        cols_df = pd.DataFrame(prof["columns"])
        st.dataframe(cols_df, use_container_width=True, hide_index=True)

        # Numerical Distributions
        if prof["numerical_stats"]:
            st.markdown("#### 🔢 Numerical Distributions & IQR Outliers")
            num_df = pd.DataFrame.from_dict(prof["numerical_stats"], orient="index")
            st.dataframe(num_df, use_container_width=True)

        # Correlations Alert
        if prof["high_correlations"]:
            st.markdown("#### 🔗 High Feature Correlations")
            corr_df = pd.DataFrame(prof["high_correlations"])
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

        # Raw preview expander
        with st.expander("👀 View Raw Sample Head (10 Rows)"):
            st.dataframe(st.session_state.dataset_df.head(10), use_container_width=True)

        # DuckDB Fast SQL Console
        with st.expander("🦆 DuckDB In-Memory SQL Console (Sub-Millisecond Columnar Queries)"):
            st.caption("Query the active dataset as `data` using standard SQL (e.g. `SELECT * FROM data LIMIT 5`).")
            sql_input = st.text_area("SQL Query", value="SELECT * FROM data LIMIT 5", height=100)
            if st.button("⚡ Run DuckDB Query"):
                try:
                    from backend.app.tools.duckdb_tool import query_duckdb
                    sql_res = query_duckdb(sql_input, st.session_state.active_dataset_path)
                    if sql_res["success"]:
                        st.success(f"Executed in **{sql_res['execution_time_ms']} ms** (Returned {len(sql_res['rows'])} of {sql_res['total_rows']:,} rows)")
                        st.dataframe(pd.DataFrame(sql_res["rows"]), use_container_width=True)
                    else:
                        st.error(f"SQL Execution Error: {sql_res['error']}")
                except Exception as e:
                    st.error(f"DuckDB Error: {e}")
    else:
        st.info("Load a dataset from the sidebar to view its automated profile.")


# --- TAB 3: Automated EDA & Executive Report ---
with tab_eda:
    st.markdown("### 📑 1-Click Executive EDA & Statistical Briefing")
    st.markdown("Autonomous multi-panel discovery scanning data health, key drivers, skewness, and generating visual dashboards.")

    if not st.session_state.active_dataset_path:
        st.info("Please load a dataset from the sidebar first.")
    else:
        if st.button("⚡ Generate Comprehensive Executive EDA Report", type="primary", use_container_width=True):
            with st.spinner("🤖 Autonomous EDA Engine discovering statistical patterns & rendering multi-panel charts..."):
                try:
                    from backend.app.graph.agents.eda_agent import generate_executive_eda
                    st.session_state.eda_report = generate_executive_eda(st.session_state.active_dataset_path)
                except Exception as e:
                    st.error(f"Error generating EDA report: {e}")

        if "eda_report" in st.session_state and st.session_state.eda_report:
            eda = st.session_state.eda_report
            st.markdown("---")

            # Quality Score Banner
            score_col1, score_col2, score_col3 = st.columns([1, 1, 2])
            score_col1.metric("🛡️ Data Quality Score", f"{eda['data_quality_score']} / 100")
            score_col2.metric("📊 Records Analyzed", f"{eda['row_count']:,}")
            score_col3.metric("📐 Feature Space", f"{eda['column_count']} Columns")

            # Key Insights Cards
            st.markdown("#### 🔍 Prioritized Statistical Discoveries")
            for ins in eda["ranked_insights"]:
                st.info(ins)

            # 4-Panel Visual Dashboard
            if eda.get("chart_path") and os.path.exists(eda["chart_path"]):
                st.markdown("#### 🖼️ Multi-Panel Executive Visual Dashboard")
                st.image(eda["chart_path"], caption="Consolidated 4-Panel Statistical Overview", use_container_width=True)

            # Executive Narrative Briefing
            st.markdown("#### 📝 Executive Narrative Briefing")
            st.markdown(eda["narrative_report"])

            # Export Section
            st.markdown("---")
            st.markdown("#### 📥 Export Publication-Grade Executive Deliverables")
            exp_col1, exp_col2 = st.columns(2)
            try:
                from backend.app.tools.export_tool import export_to_html, export_to_excel
                html_path = export_to_html(
                    report_title=f"Executive EDA: {eda['dataset_name']}",
                    narrative_text=eda["narrative_report"],
                    dataset_name=eda["dataset_name"],
                    kpis={"Data Quality Score": f"{eda['data_quality_score']}/100", "Total Records": f"{eda['row_count']:,}", "Features": eda["column_count"]},
                    chart_paths=[eda["chart_path"]] if eda.get("chart_path") else None,
                    sample_df=st.session_state.dataset_df,
                )
                with open(html_path, "rb") as hf:
                    exp_col1.download_button(
                        label="🌐 Download Interactive HTML Report",
                        data=hf.read(),
                        file_name=f"Executive_Report_{eda['dataset_name']}.html",
                        mime="text/html",
                        use_container_width=True,
                    )

                excel_path = export_to_excel(
                    report_title=f"Executive EDA: {eda['dataset_name']}",
                    dataset_name=eda["dataset_name"],
                    narrative_text=eda["narrative_report"],
                    kpis={"Data Quality Score": f"{eda['data_quality_score']}/100", "Total Records": f"{eda['row_count']:,}"},
                    num_stats=eda.get("numerical_stats"),
                    sample_df=st.session_state.dataset_df,
                )
                with open(excel_path, "rb") as xf:
                    exp_col2.download_button(
                        label="📊 Download Multi-Sheet Excel Workbook",
                        data=xf.read(),
                        file_name=f"Executive_Workbook_{eda['dataset_name']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"Export download preparation: {e}")


# --- TAB 4: Anomaly Detection ---
with tab_anomaly:
    st.markdown("### 🚨 Autonomous Anomaly & Outlier Detection")
    st.markdown("Scans dataset using machine learning to find fraudulent, extreme, or suspicious records.")

    if not st.session_state.active_dataset_path:
        st.info("Please load a dataset from the sidebar first.")
    else:
        col1, col2 = st.columns(2)
        method = col1.selectbox("Detection Algorithm", [
            ("isolation_forest", "Isolation Forest (Multivariate)"),
            ("lof", "Local Outlier Factor (Density-based)"),
            ("zscore", "Robust Z-Score / MAD (Univariate)")
        ], format_func=lambda x: x[1])[0]
        
        contamination = col2.slider("Contamination Rate (Expected Outliers %)", min_value=1, max_value=15, value=5, step=1) / 100.0
        
        if st.button("🚨 Scan for Anomalies", type="primary", use_container_width=True):
            with st.spinner(f"Running {method} scan on {st.session_state.active_dataset_name}..."):
                try:
                    from backend.app.graph.agents.anomaly_agent import generate_anomaly_report
                    st.session_state.anomaly_report = generate_anomaly_report(
                        st.session_state.active_dataset_path, 
                        method=method, 
                        contamination=contamination
                    )
                except Exception as e:
                    st.error(f"Anomaly detection error: {e}")

        if "anomaly_report" in st.session_state and st.session_state.anomaly_report:
            ar = st.session_state.anomaly_report
            st.markdown("---")
            
            # KPI Banner
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Records", f"{ar['total_records']:,}")
            k2.metric("Flagged Anomalies", f"{ar['total_anomalies']:,}", delta_color="inverse")
            k3.metric("Anomaly Rate", f"{ar['anomaly_percentage']}%")
            
            # Narrative Explanation
            st.markdown("#### 🧠 Agent Investigation Report")
            st.info(ar["narrative_report"])
            
            # Visual Scatter Plot
            if ar.get("chart_path") and os.path.exists(ar["chart_path"]):
                st.markdown("#### 📊 Anomaly Distribution Plot")
                st.image(ar["chart_path"], use_container_width=True)
                
            # Tabular Output
            if ar["total_anomalies"] > 0:
                st.markdown(f"#### ⚠️ Suspicious Records Sample (Top {len(ar['anomalous_rows'])})")
                st.dataframe(pd.DataFrame(ar["anomalous_rows"]), use_container_width=True)


# --- TAB 5: Predictive Modeling ---
with tab_predictive:
    st.markdown("### 🔮 Auto-ML Predictive Modeling & Feature Importance")
    st.markdown("Train ensemble machine learning models (Random Forest, Gradient Boosting) to predict any target column.")

    if not st.session_state.active_dataset_path or st.session_state.dataset_df is None:
        st.info("Please load a dataset from the sidebar first.")
    else:
        df_cols = st.session_state.dataset_df.columns.tolist()
        # Default selection to Survived, Sales, Profit or last column
        default_idx = 0
        for candidate in ["Survived", "Sales", "Profit", "total_amount"]:
            if candidate in df_cols:
                default_idx = df_cols.index(candidate)
                break

        col_p1, col_p2 = st.columns(2)
        target_col = col_p1.selectbox("🎯 Select Target Variable to Predict", df_cols, index=default_idx)
        model_type = col_p2.selectbox("🤖 Model Architecture", [
            ("random_forest", "Random Forest (Ensemble)"),
            ("gradient_boosting", "Gradient Boosting (Sequential)")
        ], format_func=lambda x: x[1])[0]

        if st.button("🧠 Train Predictive Model", type="primary", use_container_width=True):
            with st.spinner(f"Training Auto-ML model to predict '{target_col}'..."):
                try:
                    from backend.app.graph.agents.predictive_agent import generate_predictive_report
                    st.session_state.predictive_report = generate_predictive_report(
                        st.session_state.active_dataset_path,
                        target_column=target_col,
                        model_type=model_type
                    )
                except Exception as e:
                    st.error(f"Predictive modeling error: {e}")

        if "predictive_report" in st.session_state and st.session_state.predictive_report:
            pr = st.session_state.predictive_report
            st.markdown("---")

            # Metrics KPI Row
            st.markdown(f"#### 📊 Model Evaluation: `{pr['model_name']}` ({pr['task_type'].upper()})")
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Target Variable", pr["target_column"])
            kpi_cols[1].metric("Train / Test Split", f"{pr['train_samples']} / {pr['test_samples']}")

            if pr["task_type"] == "classification":
                kpi_cols[2].metric("Accuracy Score", f"{pr['metrics']['accuracy_pct']}%")
                kpi_cols[3].metric("Weighted F1-Score", f"{pr['metrics']['f1_score']}")
            else:
                kpi_cols[2].metric("R² Variance Explained", f"{pr['metrics']['r2_score']:.3f}")
                kpi_cols[3].metric("Root Mean Sq. Error (RMSE)", f"{pr['metrics']['rmse']:,}")

            # Feature Importance Visual
            if pr.get("chart_path") and os.path.exists(pr["chart_path"]):
                st.markdown("#### 🏆 Key Driving Factors (Feature Importance)")
                st.image(pr["chart_path"], use_container_width=True)

            # Executive Briefing
            st.markdown("#### 📝 Executive Model Briefing & Recommendations")
            st.markdown(pr["narrative_report"])

            # Sample Predictions Preview
            if pr.get("sample_predictions"):
                st.markdown("#### 🔬 Sample Test Predictions (Actual vs Predicted)")
                sample_df = pd.DataFrame(pr["sample_predictions"])
                st.dataframe(sample_df, use_container_width=True, hide_index=True)


# --- TAB 6: Business Glossary & RAG Knowledge Base ---
with tab_glossary:
    st.markdown("### 📚 Domain Business Glossary & RAG Vector Memory")
    st.markdown("ChromaDB semantic index grounding business formulas (e.g. AOV, Profit Margin, Strike Rate).")

    try:
        from backend.app.memory.vector_memory import add_glossary_term, list_all_glossary_terms, search_glossary

        search_q = st.text_input("🔍 Search Vector Knowledge Base (Semantic Similarity):", placeholder="e.g. 'How is average order value calculated?'")
        if search_q:
            results = search_glossary(search_q, top_k=3)
            st.markdown(f"**Top Matching Definitions for:** *'{search_q}'*")
            for r in results:
                st.success(f"**{r['term']}** ({r['category']}) — *Similarity: {r['similarity']:.2%}*\n\n**Definition:** {r['definition']}\n\n**Formula:** `{r['formula']}`\n\n**Pandas:** `{r['pandas_example']}`")
            st.markdown("---")

        all_terms = list_all_glossary_terms()
        if all_terms:
            st.markdown(f"#### 📖 Registered Domain Metrics ({len(all_terms)} Terms)")
            terms_df = pd.DataFrame(all_terms)[["term", "category", "formula", "definition", "pandas_example"]]
            st.dataframe(terms_df, use_container_width=True, hide_index=True)
        else:
            st.info("No glossary terms indexed yet.")

        st.markdown("---")
        with st.expander("➕ Add Custom Business Metric Definition"):
            with st.form("new_term_form"):
                new_term = st.text_input("Metric Name", placeholder="e.g. Net Churn Rate")
                new_category = st.selectbox("Category", ["Retail", "E-Commerce", "Finance", "Cricket", "Custom"])
                new_def = st.text_area("Plain English Definition", placeholder="Description of the metric...")
                new_formula = st.text_input("Mathematical Formula", placeholder="e.g. (Lost Customers / Total Customers) * 100")
                new_pandas = st.text_input("Pandas Reference Code", placeholder="e.g. (df['churned'].sum() / len(df)) * 100")
                submitted = st.form_submit_button("Index into ChromaDB")
                if submitted and new_term and new_formula:
                    add_glossary_term(new_term, new_def, new_formula, new_category, new_pandas)
                    st.success(f"Successfully indexed '{new_term}' into ChromaDB vector memory!")
                    st.rerun()

    except Exception as e:
        st.warning(f"Vector memory viewer unavailable: {e}")


# --- TAB 7: Governance & Audit Logs ---
with tab_audit:
    st.markdown("### 🛡️ Governance, Safety & Audit Trail")
    st.markdown("Immutable PostgreSQL/SQLite database tracking all agent operations, HITL approvals, latencies, and execution costs.")

    try:
        from backend.app.db.audit import get_recent_audit_logs

        if st.button("🔄 Refresh Audit Logs"):
            st.rerun()

        logs = get_recent_audit_logs(limit=30)
        if logs:
            st.markdown(f"#### 📜 Recent Session Audit Logs ({len(logs)} Events Recorded)")
            logs_df = pd.DataFrame(logs)
            st.dataframe(logs_df, use_container_width=True, hide_index=True)
        else:
            st.info("No audit log events recorded yet. Ask a question or run an analysis to generate an audit trail.")

    except Exception as e:
        st.error(f"Error loading governance audit logs: {e}")


