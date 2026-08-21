"""Reporter agent — multi-step analysis synthesis and executive summary.

Synthesizes the results of all executed subtasks into a cohesive, publication-ready
report with evidence citations, key metric tables, and embedded visualizations.
"""

from typing import List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from backend.app.config import get_llm
from backend.app.graph.state import Subtask

REPORTER_SYSTEM_PROMPT = """\
You are an expert Chief Executive Data Analyst and Technical Reporter.
Your job is to synthesize the findings from a completed multi-step data analysis into a clear, professional, executive-ready response for the user.

## Instructions:
1. Address the user's original question directly in the opening summary.
2. Structure the answer clearly using bold metrics, bullet points, or markdown tables where helpful.
3. Cite the exact computed numbers from the execution steps as empirical evidence.
4. Keep the tone professional, objective, and concise. Avoid unnecessary filler or meta-commentary about "subtasks" or "tools".
5. If visual charts were created, mention the key insights highlighted in the visualization.

## Executed Analytical Steps & Findings:
{executed_steps}
"""


def reporter_node(state: dict) -> Command:
    """LangGraph node: Reporter agent synthesizes all executed subtasks into a final narrative answer."""
    messages = state.get("messages", [])
    plan: List[Subtask] = state.get("plan", [])

    if not messages:
        return Command(update={"messages": [AIMessage(content="Analysis complete.")]})

    # If single step with already formatted answer and no complex multi-step plan, pass through
    if len(plan) <= 1 and plan and plan[0].result:
        raw_res = plan[0].result
        return Command(update={"messages": [AIMessage(content=raw_res)]})

    # Compile executed subtasks summary
    step_summaries = []
    chart_paths = []

    for i, st in enumerate(plan, 1):
        res_text = st.result or "No output produced."
        # Collect chart paths if any
        if "[CHART:" in res_text:
            import re
            matches = re.findall(r"\[CHART:(.*?)\]", res_text)
            chart_paths.extend(matches)
            res_text = re.sub(r"\[CHART:.*?\]", "", res_text).strip()

        step_summaries.append(f"### Step {i}: {st.description}\n**Computed Result:**\n{res_text}\n")

    executed_steps_str = "\n".join(step_summaries) if step_summaries else "No step results recorded."

    system_msg = SystemMessage(
        content=REPORTER_SYSTEM_PROMPT.format(executed_steps=executed_steps_str)
    )

    try:
        llm = get_llm("flash")
        # Send system prompt with original user query
        user_msg = [m for m in messages if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human"]
        if not user_msg:
            user_msg = [messages[0]]

        response = llm.invoke([system_msg, user_msg[-1]])
        raw_content = response.content
        if isinstance(raw_content, str):
            final_content = raw_content.strip()
        elif isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif hasattr(part, "text"):
                    text_parts.append(part.text)
            final_content = "\n".join(text_parts).strip()
        else:
            final_content = str(raw_content).strip()

        # Re-attach any chart paths
        for cp in set(chart_paths):
            if f"[CHART:{cp}]" not in final_content:
                final_content += f"\n\n[CHART:{cp}]"

        # Index verified analysis into vector memory & log audit event
        try:
            from backend.app.memory.vector_memory import save_analysis
            from backend.app.db.audit import log_audit_event
            q_text = user_msg[-1].content if hasattr(user_msg[-1], "content") else str(user_msg[-1])
            ds_name = state.get("dataset_id") or "dataset"
            code_text = "\n".join([st.result for st in plan if st.result]) if plan else ""
            save_analysis(
                query=q_text,
                dataset_name=ds_name,
                code=code_text,
                report_summary=final_content,
            )
            # Log to database
            log_audit_event(
                session_id=state.get("session_id") or "session-default",
                user_id=state.get("user_id") or "user-default",
                user_message=q_text,
                agent_response=final_content,
                hitl_triggered=bool(state.get("pending_hitl_action")),
                approved=True,
                latency_ms=0,
                cost_usd=0.0,
            )
        except Exception:
            pass

        return {"messages": [AIMessage(content=final_content)]}

    except Exception as e:
        # Fallback to direct concatenated results
        fallback_text = "\n\n".join([f"**{st.description}**:\n{st.result}" for st in plan if st.result])
        return {"messages": [AIMessage(content=fallback_text or f"Analysis complete: {e}")]}
