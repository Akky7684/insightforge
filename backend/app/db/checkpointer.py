"""LangGraph Checkpointer Provider.

Provides:
- PostgresSaver (production persistent state checkpointer when PostgreSQL is connected)
- MemorySaver (in-memory development checkpointer)
"""

import os
from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer():
    """Return LangGraph state checkpointer based on available environment."""
    pg_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

    if pg_url and pg_url.startswith("postgres"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            # If using psycopg connection pool
            return PostgresSaver.from_conn_string(pg_url)
        except Exception:
            pass

    # Default fallback: in-memory state checkpointer
    return MemorySaver()
