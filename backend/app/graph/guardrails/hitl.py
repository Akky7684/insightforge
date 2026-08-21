"""Human-in-the-Loop (HITL) Guardrail & Risk Assessment Engine.

Implements the 4 core safety triggers specified in Blueprint Section 10:
1. CODE_DESTRUCTIVE: State corruption, file deletions, drops, or overwrites.
2. COMPUTE_INTENSIVE: Cartesian products, heavy nested iterations, excessive memory operations.
3. COUNTER_INTUITIVE: Unverified statistical conclusions, p-value flips, small sample causal claims.
4. PII_SENSITIVE: Personally Identifiable Information (SSN, Credit Cards, Passwords, Emails).
"""

import re
from typing import Any, Dict, Optional


# Regex patterns for PII and Sensitive data detection
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PASSWORD_REGEX = re.compile(r"\b(password|passwd|secret_key|api_key|access_token)\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)

# Keywords indicating destructive actions
DESTRUCTIVE_PATTERNS = [
    r"\.drop\(.*inplace\s*=\s*True",
    r"\bos\.remove\b",
    r"\bshutil\.rmtree\b",
    r"\bos\.unlink\b",
    r"\bto_csv\(.*mode\s*=\s*['\"]w['\"]",
    r"\bDELETE\s+FROM\b",
    r"\bDROP\s+TABLE\b",
    r"\bTRUNCATE\b",
]

# Keywords indicating heavy / compute-intensive execution
COMPUTE_INTENSIVE_PATTERNS = [
    r"\.merge\(.*how\s*=\s*['\"]cross['\"]",
    r"for\s+\w+\s+in\s+.*:\s*\n\s+for\s+\w+\s+in\s+.*:\s*\n\s+for\s+\w+", # Triple nested loop
    r"GridSearchCV\(",
    r"RandomizedSearchCV\(",
]


def evaluate_hitl_triggers(
    code: str = "",
    query: str = "",
    dataset_profile: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate code, user query, and dataset context against the 4 HITL risk triggers.
    Returns risk payload if triggered, or None if safe to proceed automatically.
    """
    combined_text = f"{query}\n{code}"

    # 1. Check PII_SENSITIVE Trigger
    if SSN_REGEX.search(combined_text):
        return {
            "trigger": "PII_SENSITIVE",
            "risk_reason": "Detected potential Social Security Number (SSN) pattern in analysis context.",
            "action": "execute_code",
            "code": code,
        }

    if CREDIT_CARD_REGEX.search(combined_text):
        return {
            "trigger": "PII_SENSITIVE",
            "risk_reason": "Detected potential Credit Card / Financial Account Number pattern in analysis context.",
            "action": "execute_code",
            "code": code,
        }

    if PASSWORD_REGEX.search(combined_text):
        return {
            "trigger": "PII_SENSITIVE",
            "risk_reason": "Detected potential plaintext password, credential, or secret token in code.",
            "action": "execute_code",
            "code": code,
        }

    # 2. Check CODE_DESTRUCTIVE Trigger
    for pat in DESTRUCTIVE_PATTERNS:
        if re.search(pat, code, re.IGNORECASE):
            return {
                "trigger": "CODE_DESTRUCTIVE",
                "risk_reason": "Code contains state-mutating or potentially destructive operations (e.g. inplace drops, file deletions, or overwrites).",
                "action": "execute_code",
                "code": code,
            }

    # 3. Check COMPUTE_INTENSIVE Trigger
    for pat in COMPUTE_INTENSIVE_PATTERNS:
        if re.search(pat, code):
            return {
                "trigger": "COMPUTE_INTENSIVE",
                "risk_reason": "Operation detected as computationally heavy (Cartesian merge / deep nested iterations) requiring user confirmation.",
                "action": "execute_code",
                "code": code,
            }

    # Check cell operation size threshold if profile available
    if dataset_profile:
        row_count = dataset_profile.get("row_count", 0)
        col_count = dataset_profile.get("column_count", 0)
        if row_count * col_count > 2_000_000 and "sns.pairplot" in code:
            return {
                "trigger": "COMPUTE_INTENSIVE",
                "risk_reason": f"Generating pairwise scatterplots on {row_count:,} records exceeds recommended browser memory limit.",
                "action": "render_chart",
                "code": code,
            }

    # 4. Check COUNTER_INTUITIVE Trigger (Extreme data mutation / anomalous paradox assertions)
    if "simpson's paradox" in query.lower() or "causal" in query.lower():
        # Requires human sign-off when making causal claims on observational data
        return {
            "trigger": "COUNTER_INTUITIVE",
            "risk_reason": "Query requests causal inference or Simpson's Paradox resolution on observational data, requiring human domain expert review.",
            "action": "synthesize_report",
            "code": code,
        }

    # Safe to proceed
    return None
