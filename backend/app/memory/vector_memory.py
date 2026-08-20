"""Vector memory engine using ChromaDB for domain business glossaries and historical analysis retrieval.

Maintains two persistent collections:
1. `business_glossary`: Domain metric definitions, business rules, and calculation formulas.
2. `past_analyses`: Historical verified analytical questions, code, and executive summaries.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.app.config import get_settings

# Initialize singleton ChromaDB client
_client = None
_glossary_col = None
_analyses_col = None


def get_chroma_client():
    """Get or create persistent ChromaDB client."""
    global _client, _glossary_col, _analyses_col
    if _client is None:
        persist_dir = str(Path(get_settings().chroma_persist_dir).resolve())
        os.makedirs(persist_dir, exist_ok=True)
        _client = chromadb.PersistentClient(path=persist_dir)
        _glossary_col = _client.get_or_create_collection(
            name="business_glossary",
            metadata={"hnsw:space": "cosine"}
        )
        _analyses_col = _client.get_or_create_collection(
            name="past_analyses",
            metadata={"hnsw:space": "cosine"}
        )
        # Bootstrap default seed glossaries if empty
        if _glossary_col.count() == 0:
            bootstrap_glossaries()
    return _client


def bootstrap_glossaries(glossaries_dir: Optional[str] = None):
    """Seed business glossary collection with domain knowledge from JSON files."""
    if glossaries_dir is None:
        glossaries_dir = str(Path(get_settings().data_dir) / "glossaries")

    g_dir = Path(glossaries_dir)
    if not g_dir.exists():
        return

    client = get_chroma_client()
    col = client.get_collection("business_glossary")

    for json_file in g_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                terms = json.load(f)
                for item in terms:
                    t_id = item.get("id", str(uuid.uuid4()))
                    term = item.get("term", "")
                    definition = item.get("definition", "")
                    formula = item.get("formula", "")
                    pandas_ex = item.get("pandas_example", "")
                    category = item.get("category", "General")

                    doc_text = f"Term: {term}\nDefinition: {definition}\nFormula: {formula}\nPandas Example: {pandas_ex}\nCategory: {category}"
                    col.upsert(
                        ids=[t_id],
                        documents=[doc_text],
                        metadatas=[{
                            "term": term,
                            "definition": definition,
                            "formula": formula,
                            "pandas_example": pandas_ex,
                            "category": category,
                        }],
                    )
        except Exception as e:
            print(f"Error bootstrapping glossary file {json_file}: {e}")


def search_glossary(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Perform semantic similarity search over business glossary definitions."""
    client = get_chroma_client()
    col = client.get_collection("business_glossary")

    if col.count() == 0:
        return []

    try:
        results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
        matches = []
        if results and results.get("metadatas") and len(results["metadatas"]) > 0:
            for meta, doc, dist in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
                # Cosine distance to similarity: similarity = 1 - distance
                sim = 1.0 - dist if dist is not None else 1.0
                matches.append({
                    "term": meta.get("term"),
                    "definition": meta.get("definition"),
                    "formula": meta.get("formula"),
                    "pandas_example": meta.get("pandas_example"),
                    "category": meta.get("category"),
                    "similarity": round(float(sim), 4),
                    "document": doc,
                })
        return matches
    except Exception as e:
        print(f"Error querying glossary vector memory: {e}")
        return []


def save_analysis(
    query: str,
    dataset_name: str,
    code: str,
    report_summary: str,
    analysis_id: Optional[str] = None
) -> str:
    """Index a completed, verified analysis for historical semantic retrieval."""
    client = get_chroma_client()
    col = client.get_collection("past_analyses")

    a_id = analysis_id or f"analysis_{uuid.uuid4().hex[:8]}"
    doc_text = f"Dataset: {dataset_name}\nQuestion: {query}\nCode:\n{code}\nSummary:\n{report_summary}"

    col.upsert(
        ids=[a_id],
        documents=[doc_text],
        metadatas=[{
            "query": query,
            "dataset_name": dataset_name,
            "code": code[:1500],
            "report_summary": report_summary[:1500],
        }],
    )
    return a_id


def search_past_analyses(
    query: str,
    dataset_name: Optional[str] = None,
    top_k: int = 2
) -> List[Dict[str, Any]]:
    """Retrieve historically executed analyses similar to the active query."""
    client = get_chroma_client()
    col = client.get_collection("past_analyses")

    if col.count() == 0:
        return []

    where_clause = {"dataset_name": dataset_name} if dataset_name else None

    try:
        results = col.query(
            query_texts=[query],
            n_results=min(top_k, col.count()),
            where=where_clause,
        )
        matches = []
        if results and results.get("metadatas") and len(results["metadatas"]) > 0:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                sim = 1.0 - dist if dist is not None else 1.0
                matches.append({
                    "query": meta.get("query"),
                    "dataset_name": meta.get("dataset_name"),
                    "code": meta.get("code"),
                    "report_summary": meta.get("report_summary"),
                    "similarity": round(float(sim), 4),
                })
        return matches
    except Exception as e:
        print(f"Error querying past analyses vector memory: {e}")
        return []


def list_all_glossary_terms() -> List[Dict[str, Any]]:
    """List all registered business glossary definitions."""
    client = get_chroma_client()
    col = client.get_collection("business_glossary")

    if col.count() == 0:
        return []

    all_docs = col.get()
    terms = []
    if all_docs and all_docs.get("metadatas"):
        for m in all_docs["metadatas"]:
            terms.append(m)
    return terms


def add_glossary_term(
    term: str,
    definition: str,
    formula: str,
    category: str = "Custom",
    pandas_example: str = ""
) -> str:
    """Add a new custom glossary definition."""
    client = get_chroma_client()
    col = client.get_collection("business_glossary")

    t_id = f"custom_{uuid.uuid4().hex[:8]}"
    doc_text = f"Term: {term}\nDefinition: {definition}\nFormula: {formula}\nPandas Example: {pandas_example}\nCategory: {category}"

    col.upsert(
        ids=[t_id],
        documents=[doc_text],
        metadatas=[{
            "term": term,
            "definition": definition,
            "formula": formula,
            "pandas_example": pandas_example,
            "category": category,
        }],
    )
    return t_id
