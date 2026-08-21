"""Predictive Modeling Agent.

Coordinates Auto-ML model training (Random Forest, Gradient Boosting)
and synthesizes strategic executive briefings explaining feature importances and predictions.
"""

from typing import Any, Dict, Optional
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from backend.app.config import get_llm
from backend.app.tools.predictive_tool import train_predictive_model

PREDICTIVE_AGENT_SYSTEM_PROMPT = """\
You are an expert Chief Machine Learning Officer and Quantitative Strategist.
Your job is to transform raw Machine Learning evaluation metrics and Feature Importance rankings
into a high-impact, publication-ready Executive Predictive Modeling Briefing.

## Machine Learning Results:
Dataset: {dataset_name}
Target Variable: {target_column} ({task_type})
Model Architecture: {model_name}
Training Samples: {train_samples} | Test Samples: {test_samples}
Performance Metrics: {metrics_summary}

### Top Driving Features (Feature Importance):
{features_summary}

## Structure Your Response:
1. 📌 **Executive Model Performance Summary** (Explain how reliably the model predicts `{target_column}` using plain-English interpretations of accuracy/R²/RMSE).
2. 🔍 **Key Predictive Drivers & Influencers** (Explain the top 3-4 features that drive predictions and why they make business sense).
3. 💡 **Strategic Recommendations & Downstream Action Items** (Provide 3 actionable business strategies based on what factors influence the target).
"""


def generate_predictive_report(
    dataset_path: str,
    target_column: str,
    model_type: str = "random_forest",
) -> Dict[str, Any]:
    """Train Auto-ML model and synthesize executive narrative report."""
    results = train_predictive_model(
        dataset_path=dataset_path,
        target_column=target_column,
        model_type=model_type,
    )

    # Format metrics summary
    if results["task_type"] == "classification":
        m = results["metrics"]
        metrics_summary = f"Accuracy: {m['accuracy_pct']}% | Weighted F1-Score: {m['f1_score']} | Classes: {m['classes']}"
    else:
        m = results["metrics"]
        metrics_summary = f"R² Score: {m['r2_score']} | RMSE: {m['rmse']} | MAE: {m['mae']}"

    # Format feature importances
    features_lines = []
    for f in results["top_features"][:6]:
        features_lines.append(f"- **{f['feature']}**: {f['importance_pct']}% relative impact")
    features_summary = "\n".join(features_lines)

    # Prompt LLM for strategic narrative
    prompt = PREDICTIVE_AGENT_SYSTEM_PROMPT.format(
        dataset_name=results["dataset_name"],
        target_column=results["target_column"],
        task_type=results["task_type"].upper(),
        model_name=results["model_name"],
        train_samples=results["train_samples"],
        test_samples=results["test_samples"],
        metrics_summary=metrics_summary,
        features_summary=features_summary,
    )

    try:
        llm = get_llm("pro")
        response = llm.invoke([SystemMessage(content=prompt)])
        raw_content = response.content
        if isinstance(raw_content, str):
            narrative = raw_content.strip()
        elif isinstance(raw_content, list):
            text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content]
            narrative = " ".join(text_parts).strip()
        else:
            narrative = str(raw_content).strip()
    except Exception as e:
        narrative = f"### Predictive Model Summary\n\n**Task:** {results['task_type'].upper()} for `{target_column}`\n\n**Performance:** {metrics_summary}\n\n**Top Drivers:**\n{features_summary}\n\n*(Narrative generation note: {e})*"

    return {
        **results,
        "narrative_report": narrative,
    }


def predictive_node(state: dict) -> Command:
    """LangGraph node wrapper for predictive modeling agent."""
    dataset_path = state.get("dataset_path")
    if not dataset_path:
        return Command(
            update={"messages": [AIMessage(content="Please upload or select a dataset first!")]},
            goto=END,
        )

    # Detect target column from last user message or fallback to first numerical column
    messages = state.get("messages", [])
    last_msg = messages[-1].content if messages else ""

    # Check columns in profile
    prof = state.get("dataset_profile") or {}
    columns = [c["name"] if isinstance(c, dict) else str(c) for c in prof.get("columns", [])]
    target_col = None
    for c in columns:
        if c.lower() in last_msg.lower():
            target_col = c
            break

    if not target_col:
        # Fallback to Survived / Sales or first column
        for candidate in ["Survived", "Sales", "Profit", "total_amount", "batsman_runs"]:
            if candidate in columns:
                target_col = candidate
                break
        if not target_col and columns:
            target_col = columns[-1]

    if not target_col:
        return Command(
            update={"messages": [AIMessage(content="Unable to determine a target column for predictive modeling.")]},
            goto=END,
        )

    try:
        report = generate_predictive_report(dataset_path, target_column=target_col)
        msg = f"🔮 **Predictive Modeling Complete for `{target_col}`**\n\n{report['narrative_report']}"
        if report.get("chart_path"):
            msg += f"\n\n*(Feature importance chart saved at: `{report['chart_path']}`)*"
        return Command(
            update={
                "messages": [AIMessage(content=msg)],
            },
            goto=END,
        )
    except Exception as e:
        return Command(
            update={"messages": [AIMessage(content=f"Predictive modeling error: {e}")]},
            goto=END,
        )
