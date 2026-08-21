"""Automated EDA & Statistical Discovery Engine.

Performs:
- Comprehensive dataset health and quality assessment (Data Quality Score 0-100%).
- Distribution scanning (skewness, kurtosis, IQR outlier detection).
- Categorical cardinality and imbalance scanning.
- Key driver and correlation discovery.
- 4-panel publication-grade visual executive dashboard generation.
- Prioritized, ranked statistical insights extraction.
"""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from backend.app.config import get_settings


def generate_eda_report(dataset_path: str) -> Dict[str, Any]:
    """Analyze dataset and generate comprehensive EDA summary, insights, and multi-panel chart."""
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load data with fallback encoding
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="latin1")

    dataset_name = p.name
    total_rows = len(df)
    total_cols = len(df.columns)
    dup_rows = int(df.duplicated().sum())
    memory_mb = round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2)

    # 1. Missingness & Quality Scoring
    missing_counts = df.isnull().sum().to_dict()
    total_cells = total_rows * total_cols
    total_missing_cells = int(df.isnull().sum().sum())
    overall_missing_pct = round(float((total_missing_cells / total_cells) * 100), 2) if total_cells > 0 else 0.0

    # Data Quality Score (0 to 100)
    missing_penalty = min(overall_missing_pct * 2.0, 40.0)
    dup_penalty = min((dup_rows / total_rows * 100) * 1.5, 30.0) if total_rows > 0 else 0.0
    quality_score = max(0.0, round(100.0 - missing_penalty - dup_penalty, 1))

    # 2. Numerical Distributions
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_stats = {}
    highly_skewed = []
    outlier_counts = {}

    for c in num_cols:
        series = df[c].dropna()
        if len(series) > 0:
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            n_outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
            skew = round(float(series.skew()), 2)
            kurt = round(float(series.kurtosis()), 2) if len(series) > 3 else 0.0

            num_stats[c] = {
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "median": round(float(series.median()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "skew": skew,
                "kurtosis": kurt,
                "outlier_count": n_outliers,
            }
            outlier_counts[c] = n_outliers
            if abs(skew) >= 1.5:
                highly_skewed.append((c, skew))

    # 3. Categorical Profile
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_stats = {}
    for c in cat_cols:
        series = df[c].dropna()
        n_unique = int(series.nunique())
        top_val = str(series.mode().iloc[0]) if not series.empty else "N/A"
        top_freq = int((series == top_val).sum()) if not series.empty else 0
        top_pct = round(float((top_freq / len(series)) * 100), 1) if len(series) > 0 else 0.0

        cat_stats[c] = {
            "unique_count": n_unique,
            "top_category": top_val,
            "top_category_pct": top_pct,
            "is_high_cardinality": bool(n_unique > 50),
            "is_imbalanced": bool(top_pct >= 85.0),
        }

    # 4. Correlation Analysis
    corr_pairs = []
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1 = num_cols[i]
                col2 = num_cols[j]
                val = corr_matrix.loc[col1, col2]
                if not np.isnan(val) and abs(val) >= 0.35:
                    corr_pairs.append({
                        "var1": col1,
                        "var2": col2,
                        "correlation": round(float(val), 3),
                        "direction": "Positive" if val > 0 else "Negative",
                    })
        corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # 5. Generate Multi-Panel Visual Overview Chart (4 Panels)
    chart_dir = Path(get_settings().data_dir).parent / "outputs" / "charts"
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = f"eda_overview_{uuid.uuid4().hex[:8]}.png"
    chart_path = str((chart_dir / chart_filename).resolve())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid", palette="muted")

    # Panel 1: Primary Numerical Distribution
    primary_num = num_cols[0] if num_cols else None
    if primary_num:
        sns.histplot(df[primary_num].dropna(), kde=True, ax=axes[0, 0], color="#2563eb")
        axes[0, 0].set_title(f"Distribution: {primary_num}", fontsize=12, fontweight="bold")
    else:
        axes[0, 0].text(0.5, 0.5, "No Numerical Features", ha="center", va="center")

    # Panel 2: Top Categorical Breakdown
    primary_cat = cat_cols[0] if cat_cols else None
    if primary_cat:
        top_cats = df[primary_cat].value_counts().head(6)
        cat_labels = top_cats.index.astype(str)
        sns.barplot(x=top_cats.values, y=cat_labels, hue=cat_labels, legend=False, ax=axes[0, 1], palette="Blues_r")
        axes[0, 1].set_title(f"Top Categories: {primary_cat}", fontsize=12, fontweight="bold")
    else:
        axes[0, 1].text(0.5, 0.5, "No Categorical Features", ha="center", va="center")

    # Panel 3: Correlation Matrix Heatmap
    if len(num_cols) >= 2:
        top_num_cols = num_cols[:6]
        sub_corr = df[top_num_cols].corr()
        sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, ax=axes[1, 0], linewidths=0.5)
        axes[1, 0].set_title("Feature Correlations", fontsize=12, fontweight="bold")
    else:
        axes[1, 0].text(0.5, 0.5, "Insufficient Numerical Features", ha="center", va="center")

    # Panel 4: Group Difference Boxplot (Cat vs Num)
    if primary_cat and primary_num:
        top_groups = df[primary_cat].value_counts().head(4).index
        filtered_sub = df[df[primary_cat].isin(top_groups)]
        sns.boxplot(data=filtered_sub, x=primary_cat, y=primary_num, hue=primary_cat, legend=False, ax=axes[1, 1], palette="Set2")
        axes[1, 1].set_title(f"{primary_num} by {primary_cat}", fontsize=12, fontweight="bold")
    elif len(num_cols) >= 2:
        sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1], ax=axes[1, 1], alpha=0.6, color="#059669")
        axes[1, 1].set_title(f"{num_cols[0]} vs {num_cols[1]}", fontsize=12, fontweight="bold")
    else:
        axes[1, 1].text(0.5, 0.5, "Bivariate View", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 6. Synthesize Ranked Business Insights
    insights = []
    insights.append(f"📊 **Dataset Health**: Loaded **{total_rows:,} rows** and **{total_cols} columns** with a Data Quality Score of **{quality_score}/100** ({dup_rows} duplicates, {overall_missing_pct}% overall missingness).")

    if highly_skewed:
        top_skew = highly_skewed[0]
        insights.append(f"📈 **Distribution Skewness**: `{top_skew[0]}` exhibits strong skewness (skew = {top_skew[1]}). Recommendation: Rely on median & IQR rather than arithmetic mean.")

    if corr_pairs:
        top_corr = corr_pairs[0]
        insights.append(f"🔗 **Strongest Association**: `{top_corr['var1']}` and `{top_corr['var2']}` show a **{top_corr['direction']} correlation (r = {top_corr['correlation']})**.")

    if any(c.get("is_imbalanced") for c in cat_stats.values()):
        imbalanced = [k for k, v in cat_stats.items() if v["is_imbalanced"]][0]
        top_info = cat_stats[imbalanced]
        insights.append(f"⚖️ **Class Imbalance**: `{imbalanced}` is dominated by **'{top_info['top_category']}'** ({top_info['top_category_pct']}% of records).")

    if any(count > 0 for count in outlier_counts.values()):
        max_out_col = max(outlier_counts, key=outlier_counts.get)
        if outlier_counts[max_out_col] > 0:
            insights.append(f"🚨 **Outlier Alert**: `{max_out_col}` contains **{outlier_counts[max_out_col]} statistical outliers** beyond the 1.5x IQR boundary.")

    return {
        "dataset_name": dataset_name,
        "row_count": total_rows,
        "column_count": total_cols,
        "memory_mb": memory_mb,
        "duplicate_rows": dup_rows,
        "overall_missing_pct": overall_missing_pct,
        "data_quality_score": quality_score,
        "numerical_stats": num_stats,
        "categorical_stats": cat_stats,
        "top_correlations": corr_pairs[:5],
        "ranked_insights": insights,
        "chart_path": chart_path,
    }
