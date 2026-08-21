"""InsightForge state schema — the central data structure flowing through the graph.

This defines:
- InsightForgeState: The TypedDict that flows through every node in the LangGraph
- Subtask: A Pydantic model representing a single step in a multi-step analysis plan
"""

from typing import Annotated, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Subtask(BaseModel):
    """A single step in a multi-step analysis plan.

    Created by the Planner agent, executed by the Coder agent,
    validated by the Critic agent.
    """

    id: str = Field(description="Unique identifier for the subtask, e.g. 'task_1'")
    description: str = Field(description="What this subtask should accomplish")
    status: Literal["pending", "running", "success", "failed", "needs_approval"] = (
        Field(default="pending", description="Current execution status")
    )
    result: Optional[str] = Field(
        default=None, description="Output/result after execution"
    )
    retries: int = Field(
        default=0, description="Number of retry attempts by the Critic loop"
    )


class InsightForgeState(TypedDict):
    """Master State definition for the Multi-Agent analytical graph.
    
    Uses standard TypedDict (compatible with LangGraph Pydantic validation) and 
    handles state updates and checkpointing.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    """Chat history — automatically appended via the add_messages reducer."""

    # --- Dataset ---
    dataset_id: Optional[str]
    """Identifier for the currently active dataset (filename or UUID)."""

    dataset_path: Optional[str]
    """File path to the uploaded dataset on disk."""

    dataset_profile: Optional[dict]
    """Cached profiling summary (schema, dtypes, stats) from the Profiler agent."""

    # --- Plan ---
    plan: list[Subtask]
    """Ordered list of subtasks created by the Planner for complex queries."""

    current_subtask_idx: int
    """Index of the subtask currently being executed."""

    # --- Context & Specialized Agents ---
    rag_context: Optional[str]
    """Business glossary definitions injected by the RAG Grounding agent."""

    anomaly_results: Optional[dict]
    """Cached anomaly detection results from the Anomaly agent."""

    # --- Interaction ---
    pending_hitl_action: Optional[dict]
    """State flag indicating the workflow is paused awaiting Human-in-the-loop input."""

    # --- Telemetry ---
    session_id: str
    """Unique ID for tracing the current analytics session."""

    user_id: str
    """Identifier for the user."""
