"""Data Profiler agent — automated dataset profiling.

Runs once per new dataset:
- Schema & dtypes inspection
- Missing value metrics & null percentages
- Numerical distributions (mean, median, std, min, max, IQR outliers, skew)
- Categorical cardinality & frequency breakdown
- High correlation pair detection (|r| >= 0.5)

Caches the structured profile in InsightForgeState['dataset_profile']
to eliminate repetitive profiling tool calls by other agents.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.types import Command


def profile_dataset(file_path: str) -> Dict[str, Any]:
    """Perform comprehensive statistical and structural profiling of a CSV dataset."""
    df = pd.read_csv(file_path, encoding="latin1")
    n_rows, n_cols = df.shape

    # 1. Column Overview & Missingness
    columns_info = []
    missing_summary = {}
    total_missing_cells = int(df.isna().sum().sum())
    
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round((null_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
        dtype_str = str(df[col].dtype)
        unique_cnt = int(df[col].nunique())

        col_data = {
            "name": col,
            "dtype": dtype_str,
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_cnt,
        }
        columns_info.append(col_data)
        if null_count > 0:
            missing_summary[col] = f"{null_count} ({null_pct}%)"

    # 2. Numerical Features Profiling
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_stats = {}
    
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        numerical_stats[col] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            "median": round(float(series.median()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "skew": round(float(series.skew()), 2) if len(series) > 2 else 0.0,
            "outlier_count": outlier_count,
            "outlier_percentage": round((outlier_count / len(series)) * 100, 2),
        }

    # 3. Categorical Features Profiling
    cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    categorical_stats = {}

    for col in cat_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        top_counts = series.value_counts().head(5)
        top_breakdown = {
            str(k): int(v) for k, v in top_counts.items()
        }
        categorical_stats[col] = {
            "unique_count": int(series.nunique()),
            "top_categories": top_breakdown,
        }

    # 4. Correlation Analysis
    high_correlations = []
    if len(num_cols) >= 2:
        try:
            corr_matrix = df[num_cols].corr()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = corr_matrix.loc[c1, c2]
                    if not pd.isna(val) and abs(val) >= 0.5:
                        high_correlations.append({
                            "feature_1": c1,
                            "feature_2": c2,
                            "correlation": round(float(val), 2),
                        })
        except Exception:
            pass

    # 5. Format Text Summary (for prompt injection)
    summary_lines = [
        f"Dataset: {Path(file_path).name}",
        f"Dimensions: {n_rows:,} rows x {n_cols} columns",
        f"Total Missing Cells: {total_missing_cells:,} ({round((total_missing_cells / (n_rows * n_cols)) * 100, 2)}%)",
        f"Numerical Columns ({len(num_cols)}): {', '.join(num_cols)}",
        f"Categorical Columns ({len(cat_cols)}): {', '.join(cat_cols)}",
    ]

    if missing_summary:
        summary_lines.append(f"Missing Values by Column: {missing_summary}")

    if high_correlations:
        corr_strs = [f"{c['feature_1']} & {c['feature_2']} (r={c['correlation']})" for c in high_correlations]
        summary_lines.append(f"High Correlations (|r| >= 0.5): {', '.join(corr_strs)}")

    return {
        "dataset_name": Path(file_path).name,
        "row_count": n_rows,
        "column_count": n_cols,
        "total_missing_cells": total_missing_cells,
        "columns": columns_info,
        "numerical_stats": numerical_stats,
        "categorical_stats": categorical_stats,
        "high_correlations": high_correlations,
        "summary_text": "\n".join(summary_lines),
    }


def profiler_node(state: dict) -> dict:
    """LangGraph node: Data Profiler agent."""
    dataset_path = state.get("dataset_path")
    if not dataset_path or not Path(dataset_path).exists():
        return {"messages": [AIMessage(content="Error: Dataset file not found for profiling.")]}

    # Compute or reuse cached profile
    existing_profile = state.get("dataset_profile")
    if not existing_profile or existing_profile.get("dataset_name") != Path(dataset_path).name:
        profile = profile_dataset(dataset_path)
    else:
        profile = existing_profile

    return {"dataset_profile": profile}
