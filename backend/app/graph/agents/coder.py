"""Coder/Executor agent — writes and executes pandas/scipy code in the sandbox.

This agent receives a user question + dataset schema, generates Python code
to answer it, and executes the code via sandbox_exec. It's the workhorse
of the entire system.
"""

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command

from backend.app.config import get_llm
from backend.app.tools.sandbox_exec import execute_code_in_sandbox

CODER_SYSTEM_PROMPT = """\
You are an expert data analyst. Your job is to answer analytical questions
about a dataset by writing Python/pandas code.

## Rules
1. The dataset is already loaded as a pandas DataFrame called `df`.
2. Write concise, correct pandas/numpy/scipy/matplotlib/seaborn code to answer the question.
3. ALWAYS use `print()` to output your numerical/textual answers clearly.
4. Format numbers nicely (round to 2 decimal places where appropriate).
5. If the question asks for a chart, graph, or visualization, use matplotlib or seaborn to create it with titles and labels. Do NOT call plt.show(); the sandbox automatically captures and saves figures.
6. Do NOT import os, sys, subprocess, or any unsafe modules.
7. You CAN use: pandas, numpy, scipy, statsmodels, matplotlib, seaborn, math, statistics.
8. Keep code short and focused — no unnecessary operations.

## Dataset Info
{schema_info}

## Important
- Respond with ONLY the Python code, no markdown fences, no explanations.
- The code will be executed in a sandbox and the printed output returned.
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

        # Add full dataset row count
        full_df = pd.read_csv(dataset_path, encoding="latin1")
        lines.insert(0, f"Total rows: {len(full_df)}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error reading dataset: {e}"


def coder_node(state: dict) -> Command:
    """LangGraph node: Coder agent generates and executes code.

    Reads the latest user message, generates pandas code using Gemini,
    executes it in the sandbox, and returns the result.
    """
    messages = state["messages"]
    dataset_path = state.get("dataset_path")

    if not dataset_path:
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="No dataset loaded. Please upload a CSV file first."
                    )
                ]
            }
        )

    # Build the schema/profile-aware system prompt
    profile = state.get("dataset_profile")
    if profile and "summary_text" in profile:
        schema_info = profile["summary_text"]
    else:
        schema_info = _get_schema_info(dataset_path)

    system_msg = SystemMessage(
        content=CODER_SYSTEM_PROMPT.format(schema_info=schema_info)
    )

    # Call the LLM to generate code
    llm = get_llm("flash")
    response = llm.invoke([system_msg] + list(messages))

    # Extract text from response (handling string or list of content blocks)
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

    # Remove markdown code fences if the LLM added them
    if "```" in generated_code:
        # Extract code inside ```python ... ``` or ``` ... ```
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
            # Fallback if no block toggles were captured
            generated_code = "\n".join(l for l in generated_code.splitlines() if not l.strip().startswith("```"))

    # Execute in sandbox
    result = execute_code_in_sandbox(generated_code, dataset_path=dataset_path)

    # Build response message
    if result["success"] and result["output"]:
        answer = result["output"].strip()
        response_text = f"{answer}"
    elif result["success"]:
        response_text = "Analysis executed successfully."
    else:
        response_text = (
            f"I encountered an error while analyzing the data:\n"
            f"```\n{result['error']}\n```\n"
            f"Let me know if you would like me to try an alternative approach."
        )

    # Attach chart path token if a figure was generated
    if result.get("chart_path"):
        response_text += f"\n\n[CHART:{result['chart_path']}]"

    return Command(
        update={"messages": [AIMessage(content=response_text)]}
    )
