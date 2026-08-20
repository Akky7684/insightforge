"""InsightForge — Streamlit Frontend UI.

Interactive multi-agent data analyst interface supporting CSV/Excel uploads,
sample dataset selection, dataset inspection, and natural-language Q&A.
"""

import os
import uuid
from pathlib import Path

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
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #f0f2f6;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
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


# --- Helper Functions ---
def load_dataset(file_path: str, file_name: str):
    """Load and preview dataset in session state."""
    try:
        df = pd.read_csv(file_path, encoding="latin1")
        st.session_state.dataset_df = df
        st.session_state.active_dataset_path = str(Path(file_path).resolve())
        st.session_state.active_dataset_name = file_name
        return True
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")
        return False


def query_agent(prompt: str) -> str:
    """Send natural language query to FastAPI backend (or graph directly if backend is offline)."""
    dataset_path = st.session_state.active_dataset_path
    if not dataset_path:
        return "⚠️ Please select or upload a dataset first!"

    # Try FastAPI backend endpoint first
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": prompt,
                "dataset_path": dataset_path,
                "session_id": st.session_state.session_id,
            },
            timeout=45,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "No response received.")
    except Exception:
        # Direct graph fallback (in-process fallback if standalone)
        pass

    try:
        from langchain_core.messages import HumanMessage
        from backend.app.graph.supervisor import get_graph

        graph = get_graph()
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        state_input = {
            "messages": [HumanMessage(content=prompt)],
            "dataset_id": st.session_state.active_dataset_name,
            "dataset_path": dataset_path,
            "dataset_profile": None,
            "plan": [],
            "current_subtask_idx": 0,
            "rag_context": None,
            "pending_hitl_action": None,
            "session_id": st.session_state.session_id,
            "user_id": "user_1",
        }
        res = graph.invoke(state_input, config=config)
        ai_msgs = [m for m in res["messages"] if m.type == "ai"]
        return ai_msgs[-1].content if ai_msgs else "Analysis completed."
    except Exception as e:
        return f"❌ Execution error: {e}"


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📊 Dataset Selection")

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
                    st.success(f"Uploaded & Loaded `{uploaded_file.name}`")
                    st.session_state.messages = []
                    st.rerun()

    # If dataset is loaded, show dataset details
    if st.session_state.dataset_df is not None:
        df = st.session_state.dataset_df
        st.markdown("---")
        st.markdown(f"**Active File:** `{st.session_state.active_dataset_name}`")
        st.markdown(f"**Rows:** `{df.shape[0]:,}` | **Cols:** `{df.shape[1]}`")

        with st.expander("🔍 Column Details & Data Types"):
            dtypes_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str)})
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

        with st.expander("👀 View Data Preview (Head 5)"):
            st.dataframe(df.head(5), use_container_width=True)

        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()


# --- Main Chat Interface ---
st.markdown('<div class="main-title">⚡ InsightForge</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous Multi-Agent Data Analyst — LangGraph Orchestration & Sandboxed Python Execution</div>', unsafe_allow_html=True)

# Status bar
if st.session_state.active_dataset_name:
    st.info(f"📂 **Active Dataset:** `{st.session_state.active_dataset_name}` ({len(st.session_state.dataset_df):,} rows). Ask any analytical or statistical question below.")
else:
    # Auto-load titanic.csv as default if nothing loaded yet
    default_titanic = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"
    if default_titanic.exists():
        load_dataset(str(default_titanic), "titanic.csv")
        st.rerun()
    else:
        st.warning("👈 Please select or upload a dataset from the sidebar to begin analysis.")

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick Prompt Suggestions
if not st.session_state.messages and st.session_state.active_dataset_name == "titanic.csv":
    st.markdown("**Suggested Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    if col1.button("Average age by gender"):
        prompt_text = "What is the average age of passengers grouped by gender?"
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.spinner("Analyzing dataset in sandbox..."):
            ans = query_agent(prompt_text)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()

    if col2.button("Survival rate by passenger class"):
        prompt_text = "What was the survival rate for each passenger class (Pclass)?"
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.spinner("Analyzing dataset in sandbox..."):
            ans = query_agent(prompt_text)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()

    if col3.button("Highest fare paid"):
        prompt_text = "What was the highest fare paid and who paid it?"
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.spinner("Analyzing dataset in sandbox..."):
            ans = query_agent(prompt_text)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()

# User Input
if user_query := st.chat_input("Ask a question about the dataset (e.g., 'What is the median fare?', 'Show correlation between age and fare')..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("InsightForge agents analyzing in sandbox..."):
            response = query_agent(user_query)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
