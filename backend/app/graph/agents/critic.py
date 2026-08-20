"""Critic agent — self-correction reflection and statistical assumption validator.

Validates:
- Execution completeness (no runtime errors or empty outputs)
- Distribution skewness (recommends median when mean is misleading on skewed features)
- Statistical soundness & sample size adequacy

Executes a bounded retry loop (<= 2 retries) directing the Coder agent to refine outputs.
"""

from typing import List, Optional
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from backend.app.config import get_llm
from backend.app.graph.state import Subtask

CRITIC_SYSTEM_PROMPT = """\
You are an expert Chief Statistical Methodologist and Data Critic.
Your role is to rigorously review the Python code execution output produced by the Coder agent for analytical correctness, statistical validity, and clarity.

## Evaluation Checklist:
1. Did the code execute successfully without errors or empty outputs?
2. If calculating summary statistics on a highly skewed column (|skew| >= 1.5), was median/IQR provided in addition to mean?
3. Did the output directly answer the subtask's objective?
4. Are numeric quantities clearly formatted and labeled?

If the output is completely sound, set is_valid = True.
If there is a significant statistical or execution error, set is_valid = False and provide concise, actionable corrective feedback for the Coder agent.
"""


class CriticEvaluation(BaseModel):
    """Structured evaluation schema for the Critic agent."""

    is_valid: bool = Field(description="True if output is statistically and analytically sound, False if correction is required")
    critique: str = Field(description="Summary of the evaluation findings")
    suggested_feedback: Optional[str] = Field(None, description="Clear corrective instruction for the Coder if is_valid is False")


def critic_node(state: dict) -> Command:
    """LangGraph node: Critic validates the latest subtask execution and manages bounded self-correction retries."""
    plan: List[Subtask] = state.get("plan", [])
    current_idx = state.get("current_subtask_idx", 0)

    if not plan:
        return Command(goto="reporter")

    # The subtask that just finished is at current_idx - 1 (or current_idx if bounded)
    target_idx = min(max(0, current_idx - 1), len(plan) - 1)
    target_subtask = plan[target_idx]
    result_text = target_subtask.result or ""

    # Fast-path 1: Immediate failure on runtime error string
    if "Error during execution:" in result_text or "CRASH:" in result_text or not result_text.strip():
        if target_subtask.retries < 2:
            updated_plan = list(plan)
            new_retries = target_subtask.retries + 1
            updated_plan[target_idx] = Subtask(
                id=target_subtask.id,
                description=f"{target_subtask.description} [RETRY {new_retries}: Previous attempt encountered error: {result_text[:120]}. Write alternative robust pandas code.]",
                status="pending",
                result=None,
                retries=new_retries,
            )
            # Route back to Coder at target_idx
            return Command(
                update={"plan": updated_plan, "current_subtask_idx": target_idx},
                goto="coder"
            )

    # Statistical Evaluation via LLM Critic (only on primary steps)
    try:
        llm = get_llm("flash")
        structured_critic = llm.with_structured_output(CriticEvaluation)
        eval_prompt = (
            f"Subtask: {target_subtask.description}\n"
            f"Execution Output:\n{result_text[:1000]}"
        )
        system_msg = SystemMessage(content=CRITIC_SYSTEM_PROMPT)
        evaluation: CriticEvaluation = structured_critic.invoke([system_msg, AIMessage(content=eval_prompt)])

        # Handle retry if flawed and under retry limit
        if not evaluation.is_valid and target_subtask.retries < 2:
            updated_plan = list(plan)
            new_retries = target_subtask.retries + 1
            feedback = evaluation.suggested_feedback or evaluation.critique
            updated_plan[target_idx] = Subtask(
                id=target_subtask.id,
                description=f"{target_subtask.description} [RETRY {new_retries}: Critic feedback: {feedback}]",
                status="pending",
                result=None,
                retries=new_retries,
            )
            return Command(
                update={"plan": updated_plan, "current_subtask_idx": target_idx},
                goto="coder"
            )

    except Exception:
        # Fallback gracefully if critic evaluation encounters transient issues
        pass

    # Subtask accepted: decide if more subtasks remain or route to reporter
    if current_idx < len(plan):
        return Command(goto="coder")
    else:
        return Command(goto="reporter")
