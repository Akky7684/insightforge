"""Supervisor agent — master router for the multi-agent graph.

Routes tasks to:
- Profiler: When a dataset has not been profiled yet.
- Coder: For writing and executing analytical/statistical code.
- (Upcoming in Weeks 4-7: Planner, Critic, RAG, Reporter).
"""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from backend.app.graph.agents.coder import coder_node
from backend.app.graph.agents.profiler import profiler_node
from backend.app.graph.state import InsightForgeState


def supervisor_node(state: dict) -> Command:
    """Route user turn to the appropriate agent.

    If dataset profile is missing, route through Profiler first.
    Otherwise, route directly to Coder.
    """
    messages = state.get("messages", [])

    if not messages:
        return Command(
            update={"messages": [AIMessage(content="Hello! Please select or upload a dataset to begin.")]},
            goto=END,
        )

    # Check if dataset is provided
    if not state.get("dataset_path"):
        return Command(
            update={"messages": [AIMessage(content="Please upload or select a CSV dataset first!")]},
            goto=END,
        )

    # If dataset has not been profiled yet, route to Profiler first
    if not state.get("dataset_profile"):
        return Command(goto="profiler")

    # Otherwise route directly to Coder
    return Command(goto="coder")


def build_graph() -> StateGraph:
    """Construct and compile the multi-agent StateGraph with checkpointing."""
    graph = StateGraph(InsightForgeState)

    # Add agent nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("profiler", profiler_node)
    graph.add_node("coder", coder_node)

    # Set master entrypoint
    graph.set_entry_point("supervisor")

    # Graph edges
    graph.add_edge("profiler", "coder")
    graph.add_edge("coder", END)

    # Compile with memory checkpointer
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Singleton graph instance
_graph = None


def get_graph():
    """Get or create the compiled LangGraph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
