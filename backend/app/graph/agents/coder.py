"""Coder/Executor agent — writes and executes pandas/scipy/matplotlib code in the sandbox.

Supports both single-query execution and multi-step plan loops.
Executes code safely in the AST-hardened sandbox and records results in state['plan'].
"""

import re
from typing import List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from backend.app.config import get_llm
from backend.app.graph.state import Subtask
from backend.app.tools.sandbox_exec import execute_code_in_sandbox

CODER_SYSTEM_PROMPT = """\
You are an expert data analyst and Python programmer. Your job is to answer analytical questions
or execute a specific subtask about a dataset by writing Python/pandas/scipy/matplotlib/seaborn code.

## Rules:
1. The dataset is already loaded as a pandas DataFrame called `df`.
2. Write concise, correct code to execute the specific requested task.
3. ALWAYS use `print()` to output your numerical/textual answers clearly.
4. Format numbers nicely (round floats to 2 decimal places where appropriate).
5. If the subtask asks for a chart, graph, or visualization, use matplotlib or seaborn with titles and labels. Do NOT call plt.show(); the sandbox automatically captures and saves all open figures.
6. Do NOT import os, sys, subprocess, or any unsafe modules.
7. Allowed packages: pandas, numpy, scipy, statsmodels, matplotlib, seaborn, math, statistics, collections, itertools.
8. Respond with ONLY executable Python code inside ```python ... ``` fences or as raw code without conversational text.

## Dataset Profile:
{schema_info}

## Context from Prior Steps (if any):
{prior_context}
"""


def _get_schema_info(dataset_path: str) -> str:
    """Generate a schema summary string for the LLM prompt."""
    import pandas as pd

    try:
        df = pd.read_csv(dataset_path, encoding="latin1", nrows=5)
        lines = [
            f"Columns ({len(df.columns)}): {', '.join(df.columns.tolist())}",
            f"\nData types:\n{df.dtypes.to_string()}",
            f"\nFirst 3 rows:\n{df.head(3).to_string()}",
        ]
        full_df = pd.read_csv(dataset_path, encoding="latin1")
        lines.insert(0, f"Total rows: {len(full_df)}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading dataset: {e}"


def coder_node(state: dict) -> Command:
    """LangGraph node: Coder agent generates and executes code for the active subtask or prompt."""
    messages = state.get("messages", [])
    dataset_path = state.get("dataset_path")

    if not dataset_path:
        return Command(
            update={"messages": [AIMessage(content="No dataset loaded. Please upload a CSV file first.")]},
            goto="reporter"
        )

    # Determine active task and prior context
    plan: List[Subtask] = state.get("plan", [])
    current_idx: int = state.get("current_subtask_idx", 0)

    if plan and current_idx < len(plan):
        active_subtask = plan[current_idx]
        task_prompt = f"Execute this step: {active_subtask.description}"
        # Build prior context
        prior_steps = [
            f"- Step {i+1} ({plan[i].description}): {plan[i].result}"
            for i in range(current_idx)
            if plan[i].result
        ]
        prior_context = "\n".join(prior_steps) if prior_steps else "None (first step)."
    else:
        user_query = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        task_prompt = user_query
        prior_context = "None."

    # Build schema/profile-aware prompt
    profile = state.get("dataset_profile")
    if profile and "summary_text" in profile:
        schema_info = profile["summary_text"]
    else:
        schema_info = _get_schema_info(dataset_path)

    system_msg = SystemMessage(
        content=CODER_SYSTEM_PROMPT.format(
            schema_info=schema_info,
            prior_context=prior_context,
        )
    )

    # Call Gemini LLM
    llm = get_llm("flash")
    prompt_msgs = [system_msg, HumanMessage(content=task_prompt)]
    response = llm.invoke(prompt_msgs)

    # Extract clean code
    raw_content = response.content
    if isinstance(raw_content, str):
        generated_code = raw_content.strip()
    elif isinstance(raw_content, list):
        text_parts = []
        for part in raw_content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif hasattr(part, "text"):
                text_parts.append(part.text)
        generated_code = "\n".join(text_parts).strip()
    else:
        generated_code = str(raw_content).strip()

    # Extract code from fences if present
    if "```" in generated_code:
        lines = []
        in_block = False
        for line in generated_code.splitlines():
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                lines.append(line)
        if lines:
            generated_code = "\n".join(lines)
        else:
            generated_code = "\n".join(l for l in generated_code.splitlines() if not l.strip().startswith("```"))

    # Execute in AST-hardened sandbox
    result = execute_code_in_sandbox(generated_code, dataset_path=dataset_path)

    # Format result text
    if result["success"] and result["output"]:
        result_text = result["output"].strip()
    elif result["success"]:
        result_text = "Analysis executed successfully."
    else:
        result_text = f"Error during execution: {result['error']}"

    if result.get("chart_path"):
        result_text += f"\n\n[CHART:{result['chart_path']}]"

    # Update plan in state
    updated_plan = list(plan)
    next_idx = current_idx + 1

    if updated_plan and current_idx < len(updated_plan):
        updated_plan[current_idx] = Subtask(
            id=updated_plan[current_idx].id,
            description=updated_plan[current_idx].description,
            status="success" if result["success"] else "failed",
            result=result_text,
            retries=updated_plan[current_idx].retries,
        )
    else:
        updated_plan = [
            Subtask(
                id="task_1",
                description=task_prompt,
                status="success" if result["success"] else "failed",
                result=result_text,
            )
        ]

    # Forward to Critic agent for validation and reflection
    return Command(
        update={
            "plan": updated_plan,
            "current_subtask_idx": next_idx,
        },
        goto="critic"
    )
