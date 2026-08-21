"""Anomaly Detection Agent.

Executes the Anomaly Detection Engine (Isolation Forest, LOF, Z-Score) and 
synthesizes a plain-English narrative explaining why specific rows were flagged.
"""

from typing import Any, Dict
from langchain_core.messages import SystemMessage

from backend.app.config import get_llm
from backend.app.tools.anomaly_tool import detect_anomalies

ANOMALY_SYSTEM_PROMPT = """\
You are an expert Forensic Data Analyst and Risk Investigator.
Your job is to analyze the output of an automated Anomaly Detection algorithm (like Isolation Forest) 
and explain in plain English *why* these specific records were flagged as anomalies or outliers.

## Raw Anomaly Output:
Dataset: {dataset_name}
Method: {method} (Contamination: {contamination}%)
Features Analyzed: {features}
Total Anomalies Found: {total_anomalies} out of {total_records} ({anomaly_percentage}%)

### Sample of Anomalous Rows Flagged:
{anomalous_rows_json}

## Instructions for your Narrative Report:
1. **Summary**: Briefly state how many anomalies were found and what algorithm was used.
2. **Key Outlier Patterns**: Look at the sample of anomalous rows provided. Explain *what makes them weird*. 
   - Are there extreme values in a specific column? 
   - Is there a strange combination (e.g. high sales but negative profit)?
3. **Business Impact**: Suggest what these anomalies might represent (e.g., fraud, data entry errors, VIP customers, system glitches) and what the user should do next.
"""

def generate_anomaly_report(
    dataset_path: str, 
    method: str = "isolation_forest", 
    contamination: float = 0.05
) -> Dict[str, Any]:
    """Execute anomaly detection and synthesize a narrative report explaining the outliers."""
    # 1. Run detection engine
    results = detect_anomalies(dataset_path, method, contamination)
    
    if results["total_anomalies"] == 0:
        return {
            **results,
            "narrative_report": "No significant anomalies or outliers were detected in this dataset using the specified parameters."
        }
    
    # 2. Format row sample for LLM
    import json
    rows_str = json.dumps(results["anomalous_rows"][:10], indent=2)  # Give top 10 for reasoning
    
    prompt = ANOMALY_SYSTEM_PROMPT.format(
        dataset_name=results["dataset_name"],
        method=results["method"],
        contamination=results["contamination"] * 100,
        features=", ".join(results["features_analyzed"]),
        total_anomalies=results["total_anomalies"],
        total_records=results["total_records"],
        anomaly_percentage=results["anomaly_percentage"],
        anomalous_rows_json=rows_str
    )
    
    # 3. Synthesize narrative
    try:
        llm = get_llm("pro")  # Reasoning model preferred for pattern analysis
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
        narrative = f"*(Narrative generation failed: {e})*\n\nReview the table of anomalies below."

    return {
        **results,
        "narrative_report": narrative
    }


def anomaly_node(state: dict):
    """LangGraph node wrapper for the anomaly detection agent."""
    dataset_path = state.get("dataset_path")
    if not dataset_path:
        from langchain_core.messages import AIMessage
        from langgraph.types import Command
        from langgraph.graph import END
        return Command(
            update={"messages": [AIMessage(content="No dataset available for anomaly detection.")]},
            goto=END
        )
        
    res = generate_anomaly_report(dataset_path)
    
    narrative = res.get("narrative_report", "Anomaly scan complete.")
    chart_path = res.get("chart_path")
    
    from langchain_core.messages import AIMessage
    from langgraph.types import Command
    from langgraph.graph import END
    
    msg_content = f"🚨 **Anomaly Detection Complete**\n\n{narrative}"
    if chart_path:
        msg_content += f"\n\n*(Visual anomaly plot generated at: `{chart_path}`)*"
        
    return Command(
        update={
            "anomaly_results": res,
            "messages": [AIMessage(content=msg_content)]
        },
        goto=END
    )
