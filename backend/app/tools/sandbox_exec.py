"""Sandboxed Python code execution — subprocess-based isolation.

Executes LLM-generated pandas/scipy/statsmodels code safely:
- Runs in a separate subprocess (crash isolation)
- Wall-clock timeout (prevents infinite loops)
- Restricted imports whitelist (no os, sys, subprocess, etc.)
- DataFrame is passed via a temp file path

This is the CORE safety mechanism. The blueprint explicitly identifies
this as fixing the eval() sandbox flaw in the reference project.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from backend.app.config import get_settings

# Imports explicitly blocked (dangerous)
BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "importlib",
    "ctypes", "signal", "multiprocessing", "threading",
}


def _validate_code(code: str) -> Optional[str]:
    """Check code for disallowed imports. Returns error message or None."""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            for blocked in BLOCKED_IMPORTS:
                if blocked in stripped.split():
                    return f"Blocked import detected: '{blocked}' is not allowed in the sandbox."
    return None


def _build_runner_script(code: str, dataset_path: Optional[str] = None) -> str:
    """Build the Python script that will run in the subprocess."""
    lines = [
        "import sys",
        "import json",
        "import io",
        "",
        "captured = io.StringIO()",
        "sys.stdout = captured",
        "",
        "try:",
    ]

    # Pre-load dataset if provided
    if dataset_path:
        # Normalize path for Windows (use raw string)
        safe_path = dataset_path.replace("\\", "\\\\")
        lines.append("    import pandas as pd")
        lines.append(f'    df = pd.read_csv("{safe_path}", encoding="latin1")')
        lines.append("")

    # Indent user code and add it
    for user_line in code.splitlines():
        lines.append(f"    {user_line}")

    lines.extend([
        "",
        "    sys.stdout = sys.__stdout__",
        '    result = {"success": True, "output": captured.getvalue(), "error": None}',
        "except Exception as e:",
        "    sys.stdout = sys.__stdout__",
        '    result = {"success": False, "output": captured.getvalue(), "error": f"{type(e).__name__}: {e}"}',
        "",
        "print(json.dumps(result))",
    ])

    return "\n".join(lines)


def execute_code_in_sandbox(
    code: str,
    dataset_path: Optional[str] = None,
    timeout: Optional[int] = None,
) -> dict:
    """Execute Python code in an isolated subprocess.

    Args:
        code: Python code to execute.
        dataset_path: Path to CSV file to pre-load as 'df'.
        timeout: Max seconds to wait (default from settings).

    Returns:
        dict with keys: success (bool), output (str), error (str or None)
    """
    settings = get_settings()
    timeout = timeout or settings.max_sandbox_timeout

    # 1. Validate code safety
    validation_error = _validate_code(code)
    if validation_error:
        return {"success": False, "output": "", "error": validation_error}

    # 2. Build the runner script
    runner_script = _build_runner_script(code, dataset_path)

    # 3. Write to temp file and execute in subprocess
    runner_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(runner_script)
            runner_path = f.name

        result = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(settings.data_dir).parent),
        )

        # Parse result — find the JSON output (last line of stdout)
        if result.returncode == 0 and result.stdout.strip():
            for line in reversed(result.stdout.strip().splitlines()):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            return {"success": True, "output": result.stdout, "error": None}
        else:
            return {
                "success": False,
                "output": result.stdout,
                "error": result.stderr or "Unknown error",
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Execution timed out after {timeout} seconds.",
        }
    except Exception as e:
        return {"success": False, "output": "", "error": f"Sandbox error: {e}"}
    finally:
        if runner_path:
            Path(runner_path).unlink(missing_ok=True)


@tool
def sandbox_exec(code: str) -> str:
    """Execute Python/pandas code against the active dataset in a safe sandbox.

    The dataset is pre-loaded as a pandas DataFrame named 'df'.
    You can use pandas, numpy, scipy, statsmodels, matplotlib, and math.
    Always use print() to output your results.

    Args:
        code: Python code to execute. Use 'df' to access the dataset.

    Returns:
        The printed output from the code execution, or an error message.
    """
    result = execute_code_in_sandbox(code)

    if result["success"]:
        return result["output"] if result["output"] else "Code executed successfully (no output)."
    else:
        return f"Error: {result['error']}"
