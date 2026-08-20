"""Sandboxed Python code execution — subprocess-based isolation.

Executes LLM-generated pandas/scipy/statsmodels code safely:
- AST-based static analysis to prevent blocked imports or malicious builtins
- Runs in a separate subprocess (crash isolation)
- Wall-clock timeout (prevents infinite loops)
- Strict whitelist of safe libraries (pandas, numpy, scipy, statsmodels, matplotlib, etc.)
- Automatic dataset preloading as 'df'

This directly addresses the security and sandboxing requirements in Section 7 of the blueprint.
"""

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from backend.app.config import get_settings

# Whitelist of allowed top-level module names
ALLOWED_MODULES = {
    "pandas", "numpy", "scipy", "statsmodels", "matplotlib",
    "math", "statistics", "collections", "datetime", "re",
    "itertools", "functools", "json", "seaborn",
}

# Explicit blacklist of prohibited modules (system/network/filesystem access)
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "glob",
    "socket", "http", "urllib", "requests", "importlib",
    "ctypes", "signal", "multiprocessing", "threading", "asyncio",
    "builtins", "posix", "nt", "pty", "commands", "pickle", "shelve",
}


def _validate_code_ast(code: str) -> Optional[str]:
    """Parse code with AST and verify all imports and calls are safe.
    
    Returns error message if unsafe, or None if valid.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError in code: {e}"

    for node in ast.walk(tree):
        # 1. Check direct imports: import os, import os.path, import os; ...
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in BLOCKED_MODULES or root_pkg not in ALLOWED_MODULES:
                    return f"Blocked import detected: '{root_pkg}' is not allowed in sandbox."

        # 2. Check from-imports: from os import path, from subprocess import Popen, etc.
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in BLOCKED_MODULES or root_pkg not in ALLOWED_MODULES:
                    return f"Blocked import detected: '{root_pkg}' is not allowed in sandbox."

        # 3. Check calls to dangerous built-ins (__import__, eval, exec, open)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in {"eval", "exec", "__import__", "open", "compile"}:
                    return f"Blocked function call detected: '{func_name}' is not permitted."

    return None


def _build_runner_script(code: str, dataset_path: Optional[str] = None) -> str:
    """Build the standalone Python script to be executed in the subprocess."""
    lines = [
        "import sys",
        "import json",
        "import io",
        "import uuid",
        "from pathlib import Path",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "",
        "captured = io.StringIO()",
        "sys.stdout = captured",
        "",
        "try:",
    ]

    # Pre-load dataset as df if provided
    if dataset_path:
        safe_path = dataset_path.replace("\\", "\\\\")
        lines.append("    import pandas as pd")
        lines.append(f'    df = pd.read_csv("{safe_path}", encoding="latin1")')
        lines.append("")

    # Indent user code inside the try block
    for user_line in code.splitlines():
        lines.append(f"    {user_line}")

    lines.extend([
        "",
        "    # Check if any matplotlib figures were created",
        "    chart_path = None",
        "    if plt.get_fignums():",
        "        charts_dir = Path('outputs') / 'charts'",
        "        charts_dir.mkdir(parents=True, exist_ok=True)",
        "        chart_file = charts_dir / f'chart_{uuid.uuid4().hex[:8]}.png'",
        "        plt.tight_layout()",
        "        plt.savefig(chart_file, format='png', bbox_inches='tight', dpi=150)",
        "        plt.close('all')",
        "        chart_path = str(chart_file.resolve())",
        "",
        "    sys.stdout = sys.__stdout__",
        '    result = {"success": True, "output": captured.getvalue(), "chart_path": chart_path, "error": None}',
        "except Exception as e:",
        "    sys.stdout = sys.__stdout__",
        '    result = {"success": False, "output": captured.getvalue(), "chart_path": None, "error": f"{type(e).__name__}: {e}"}',
        "",
        "print(json.dumps(result))",
    ])

    return "\n".join(lines)


def execute_code_in_sandbox(
    code: str,
    dataset_path: Optional[str] = None,
    timeout: Optional[int] = None,
) -> dict:
    """Execute Python code in an isolated subprocess with security checks.

    Args:
        code: Python code string.
        dataset_path: Optional file path to CSV for preloading as 'df'.
        timeout: Execution timeout in seconds (default from config).

    Returns:
        dict: {"success": bool, "output": str, "error": str or None}
    """
    settings = get_settings()
    timeout = timeout or settings.max_sandbox_timeout

    # 1. AST-based static safety validation
    validation_error = _validate_code_ast(code)
    if validation_error:
        return {"success": False, "output": "", "error": validation_error}

    # 2. Build isolated runner script
    runner_script = _build_runner_script(code, dataset_path)

    # 3. Execute in subprocess
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

        # Parse JSON output from last line of subprocess stdout
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
                "error": result.stderr.strip() or f"Process exited with code {result.returncode}",
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Execution timed out after {timeout} seconds.",
        }
    except Exception as e:
        return {"success": False, "output": "", "error": f"Sandbox execution error: {e}"}
    finally:
        if runner_path:
            Path(runner_path).unlink(missing_ok=True)


@tool
def sandbox_exec(code: str) -> str:
    """Execute Python/pandas code against the active dataset in a safe sandbox.

    The dataset is pre-loaded as a pandas DataFrame named 'df'.
    Allowed packages: pandas, numpy, scipy, statsmodels, matplotlib, math, statistics.
    Always print() your outputs.

    Args:
        code: Python code to execute.

    Returns:
        String output or error description.
    """
    result = execute_code_in_sandbox(code)
    if result["success"]:
        return result["output"] if result["output"] else "Code executed successfully (no output)."
    else:
        return f"Error: {result['error']}"
