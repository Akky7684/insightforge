"""Governance & Audit Logging Service.

Records all multi-agent executions, HITL approvals/rejections, latencies, and costs into the database.
"""

from typing import Any, Dict, List, Optional
from backend.app.db.database import get_db_connection


def log_audit_event(
    session_id: str,
    user_id: str,
    user_message: str,
    agent_response: str,
    hitl_triggered: bool = False,
    hitl_trigger_type: Optional[str] = None,
    hitl_risk_reason: Optional[str] = None,
    approved: bool = True,
    latency_ms: int = 0,
    cost_usd: float = 0.0,
) -> int:
    """Log an interaction or governance event to the audit_logs table."""
    conn, dialect = get_db_connection()
    cursor = conn.cursor()

    if dialect == "postgres":
        insert_sql = """
        INSERT INTO audit_logs (
            session_id, user_id, user_message, agent_response,
            hitl_triggered, hitl_trigger_type, hitl_risk_reason,
            approved, latency_ms, cost_usd
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        cursor.execute(insert_sql, (
            session_id, user_id, user_message, agent_response,
            hitl_triggered, hitl_trigger_type, hitl_risk_reason,
            approved, latency_ms, cost_usd
        ))
        event_id = cursor.fetchone()[0]
    else:
        insert_sql = """
        INSERT INTO audit_logs (
            session_id, user_id, user_message, agent_response,
            hitl_triggered, hitl_trigger_type, hitl_risk_reason,
            approved, latency_ms, cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        cursor.execute(insert_sql, (
            session_id, user_id, user_message, agent_response,
            1 if hitl_triggered else 0, hitl_trigger_type, hitl_risk_reason,
            1 if approved else 0, latency_ms, cost_usd
        ))
        event_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return event_id


def get_recent_audit_logs(limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve recent governance audit logs sorted by timestamp descending."""
    conn, dialect = get_db_connection()
    cursor = conn.cursor()

    query_sql = f"""
    SELECT id, session_id, user_id, user_message, agent_response,
           hitl_triggered, hitl_trigger_type, hitl_risk_reason,
           approved, latency_ms, cost_usd, created_at
    FROM audit_logs
    ORDER BY id DESC
    LIMIT {int(limit)};
    """

    cursor.execute(query_sql)
    rows = cursor.fetchall()

    results = []
    for r in rows:
        if dialect == "postgres":
            results.append(dict(r))
        else:
            results.append({
                "id": r["id"],
                "session_id": r["session_id"],
                "user_id": r["user_id"],
                "user_message": r["user_message"],
                "agent_response": r["agent_response"],
                "hitl_triggered": bool(r["hitl_triggered"]),
                "hitl_trigger_type": r["hitl_trigger_type"],
                "hitl_risk_reason": r["hitl_risk_reason"],
                "approved": bool(r["approved"]),
                "latency_ms": r["latency_ms"],
                "cost_usd": float(r["cost_usd"]),
                "created_at": str(r["created_at"]),
            })

    conn.close()
    return results
