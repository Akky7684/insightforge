"""Database Connection & Schema Manager.

Supports:
- PostgreSQL (via psycopg2 / asyncpg when DATABASE_URL or POSTGRES_URL is configured)
- SQLite (zero-config, zero-cost local embedded fallback stored in `./insightforge.db`)
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.config import get_settings

DB_PATH = Path(get_settings().data_dir).parent / "insightforge.db"


def get_db_connection():
    """Get database connection (SQLite by default, or PostgreSQL if configured)."""
    pg_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

    if pg_url and pg_url.startswith("postgres"):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn = psycopg2.connect(pg_url)
            return conn, "postgres"
        except Exception:
            pass  # Fall back to SQLite if PostgreSQL connection fails

    # Default: Embedded SQLite
    conn = sqlite3.connect(str(DB_PATH.resolve()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


def init_db():
    """Initialize database tables including audit_logs and governance tracking."""
    conn, dialect = get_db_connection()
    cursor = conn.cursor()

    if dialect == "postgres":
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(64),
            user_id VARCHAR(64),
            user_message TEXT,
            agent_response TEXT,
            hitl_triggered BOOLEAN DEFAULT FALSE,
            hitl_trigger_type VARCHAR(64),
            hitl_risk_reason TEXT,
            approved BOOLEAN DEFAULT TRUE,
            latency_ms INT DEFAULT 0,
            cost_usd NUMERIC(8, 6) DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    else:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id TEXT,
            user_message TEXT,
            agent_response TEXT,
            hitl_triggered INTEGER DEFAULT 0,
            hitl_trigger_type TEXT,
            hitl_risk_reason TEXT,
            approved INTEGER DEFAULT 1,
            latency_ms INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()


# Auto-initialize database on import
init_db()
