"""RAG Grounding agent — semantic domain retrieval from ChromaDB business glossary.

Intercepts the incoming analytical prompt, retrieves relevant domain metric definitions,
formulas, and past analyses, and injects them into state['rag_context'].
"""

from typing import Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage

from backend.app.memory.vector_memory import search_glossary, search_past_analyses


def rag_node(state: dict) -> dict:
    """LangGraph node: RAG Grounding agent retrieves domain glossary definitions and past analyses."""
    messages = state.get("messages", [])
    if not messages:
        return {"rag_context": None}

    # Extract user prompt text
    user_msg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    dataset_name = state.get("dataset_id") or "dataset"

    # 1. Semantic search over domain business glossary
    glossary_matches = search_glossary(user_msg, top_k=2)

    # 2. Semantic search over historically executed verified analyses
    past_matches = search_past_analyses(user_msg, dataset_name=dataset_name, top_k=1)

    rag_blocks = []

    # Filter glossary matches by relevance threshold
    valid_glossary = [g for g in glossary_matches if g.get("similarity", 0) >= 0.35]
    if valid_glossary:
        rag_blocks.append("### Domain Business Glossary & Calculation Rules:")
        for item in valid_glossary:
            term = item.get("term")
            defn = item.get("definition")
            formula = item.get("formula")
            p_ex = item.get("pandas_example")
            rag_blocks.append(f"- **{term}**: {defn}\n  - Formula: `{formula}`\n  - Reference Code: `{p_ex}`")

    # Add past verified analysis if relevant
    valid_past = [p for p in past_matches if p.get("similarity", 0) >= 0.60]
    if valid_past:
        rag_blocks.append("\n### Verified Past Analysis Pattern:")
        for p in valid_past:
            rag_blocks.append(f"- Prior Question: {p.get('query')}\n  - Reference Logic: {p.get('code')[:200]}...")

    rag_context_str = "\n".join(rag_blocks) if rag_blocks else None

    return {"rag_context": rag_context_str}
