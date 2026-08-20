"""Supervisor agent — master router for the multi-agent graph.

Coordinates:
- Profiler: Automated schema & distribution profiling.
- RAG Agent: Domain business glossary grounding & memory retrieval.
- Planner: Multi-step task decomposition into Subtasks.
- Coder: Sandboxed execution of subtasks.
- Critic: Statistical validation & bounded self-correction loop (<= 2 retries).
- Reporter: Executive synthesis & long-term memory persistence.
"""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from backend.app.graph.agents.coder import coder_node
from backend.app.graph.agents.critic import critic_node
from backend.app.graph.agents.planner import planner_node
from backend.app.graph.agents.profiler import profiler_node
from backend.app.graph.agents.rag_agent import rag_node
from backend.app.graph.agents.reporter import reporter_node
from backend.app.graph.state import InsightForgeState


def supervisor_node(state: dict) -> Command:
    """Route user turn through Profiler or RAG Grounding."""
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

    # Otherwise route to RAG agent for domain grounding
    return Command(goto="rag")


def build_graph() -> StateGraph:
    """Construct and compile the multi-agent StateGraph with checkpointing."""
    graph = StateGraph(InsightForgeState)

    # Add agent nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("profiler", profiler_node)
    graph.add_node("rag", rag_node)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reporter", reporter_node)

    # Set master entrypoint
    graph.set_entry_point("supervisor")

    # Static edges
    graph.add_edge("profiler", "rag")
    graph.add_edge("rag", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("reporter", END)

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
