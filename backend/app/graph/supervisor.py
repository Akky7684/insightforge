"""Supervisor agent — classifies intent and routes to specialized agents.

For Week 2, this is a simple router that sends everything to the Coder.
In later weeks, it will route to Planner, Profiler, RAG, etc.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.app.graph.state import InsightForgeState
from backend.app.graph.agents.coder import coder_node


def supervisor_node(state: dict) -> Command:
    """Route the user's message to the appropriate agent.

    Week 2: Always routes to the coder agent.
    Future weeks: Will classify intent and route to Planner, Profiler, etc.
    """
    messages = state.get("messages", [])

    if not messages:
        return Command(
            update={
                "messages": [AIMessage(content="Hello! Please ask me a question about your dataset.")]
            },
            goto=END,
        )

    # Check if dataset is loaded
    if not state.get("dataset_path"):
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="Please upload a CSV dataset first, then ask me a question about it!"
                    )
                ]
            },
            goto=END,
        )

    # Week 2: Route everything to the coder
    return Command(goto="coder")


def build_graph() -> StateGraph:
    """Build and compile the InsightForge LangGraph.

    Returns a compiled graph ready to be invoked.
    """
    # Create the graph with our state schema
    graph = StateGraph(InsightForgeState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("coder", coder_node)

    # Set entry point
    graph.set_entry_point("supervisor")

    # Add edges: coder always returns to END after execution
    graph.add_edge("coder", END)

    # Compile with memory checkpointer (SQLite-based, in-memory for dev)
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled


# Singleton graph instance
_graph = None


def get_graph():
    """Get or create the compiled LangGraph (singleton)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
