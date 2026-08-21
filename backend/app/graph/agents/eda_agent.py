"""Automated EDA & Insights Agent — generates executive data health summaries and strategic recommendations.

Coordinates automated statistical discovery, multi-panel visual generation,
and narrative synthesis for 1-click comprehensive dataset briefings.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from langchain_core.messages import AIMessage, SystemMessage

from backend.app.config import get_llm
from backend.app.tools.eda_tool import generate_eda_report

EDA_AGENT_SYSTEM_PROMPT = """\
You are an expert Chief Data Scientist and Strategic Analytics Officer.
Your job is to transform raw automated statistical scan results into a high-impact, publication-ready Executive EDA Briefing.

## Structure Your Response:
1. 📌 **Executive Summary & Data Quality Assessment** (Highlight the overall health score, dimensions, and cleanliness).
2. 🔍 **Key Statistical Findings & Driver Patterns** (Cite exact numbers, top correlations, and primary distribution highlights).
3. ⚠️ **Data Risk & Outlier Flags** (Identify severe skewness where median must be used, outlier counts, and class imbalances).
4. 💡 **Strategic Recommendations & High-Value Next Steps** (Suggest 3 actionable downstream analyses, segmentation targets, or ML opportunities).

## Raw Statistical Discovery Data:
{eda_summary_text}
"""


def generate_executive_eda(dataset_path: str) -> Dict[str, Any]:
    """Execute automated EDA engine and synthesize findings into an executive report with visuals."""
    eda_data = generate_eda_report(dataset_path)

    # Format summary text for LLM
    summary_lines = [
        f"Dataset: {eda_data['dataset_name']}",
        f"Rows: {eda_data['row_count']:,} | Columns: {eda_data['column_count']}",
        f"Data Quality Score: {eda_data['data_quality_score']}/100",
        f"Duplicate Rows: {eda_data['duplicate_rows']} | Overall Missingness: {eda_data['overall_missing_pct']}%",
        "\nTop Ranked Insights:",
    ]
    for ins in eda_data["ranked_insights"]:
        summary_lines.append(f"- {ins}")

    if eda_data["top_correlations"]:
        summary_lines.append("\nTop Feature Correlations:")
        for corr in eda_data["top_correlations"]:
            summary_lines.append(f"- {corr['var1']} vs {corr['var2']}: r = {corr['correlation']} ({corr['direction']})")

    eda_summary_text = "\n".join(summary_lines)

    # Synthesize narrative with Gemini
    try:
        llm = get_llm("flash")
        system_msg = SystemMessage(
            content=EDA_AGENT_SYSTEM_PROMPT.format(eda_summary_text=eda_summary_text)
        )
        response = llm.invoke([system_msg])
        raw_content = response.content
        if isinstance(raw_content, str):
            narrative = raw_content.strip()
        elif isinstance(raw_content, list):
            text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content]
            narrative = " ".join(text_parts).strip()
        else:
            narrative = str(raw_content).strip()
    except Exception as e:
        narrative = f"### Automated EDA Summary\n\n{eda_summary_text}\n\n*(Narrative generation note: {e})*"

    return {
        "dataset_name": eda_data["dataset_name"],
        "data_quality_score": eda_data["data_quality_score"],
        "row_count": eda_data["row_count"],
        "column_count": eda_data["column_count"],
        "ranked_insights": eda_data["ranked_insights"],
        "top_correlations": eda_data["top_correlations"],
        "chart_path": eda_data["chart_path"],
        "narrative_report": narrative,
        "numerical_stats": eda_data["numerical_stats"],
        "categorical_stats": eda_data["categorical_stats"],
    }
