"""Chart Generation Tool — structured specification-driven plotting engine.

Generates publication-ready matplotlib / seaborn charts from structured Pydantic specifications
and saves them to outputs/charts/ with unique identifiers.
"""

import base64
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field

from backend.app.config import get_settings


class ChartSpec(BaseModel):
    """Specification for generating a structured chart."""

    chart_type: Literal["bar", "line", "scatter", "histogram", "box", "heatmap", "pie", "countplot"] = Field(
        ..., description="Type of visualization"
    )
    title: str = Field(..., description="Chart title")
    x: Optional[str] = Field(None, description="Column name for X-axis")
    y: Optional[str] = Field(None, description="Column name for Y-axis")
    hue: Optional[str] = Field(None, description="Column name for grouping/color segmentation")
    aggregation: Optional[Literal["mean", "sum", "count", "median"]] = Field(
        None, description="Aggregation method when grouping"
    )
    xlabel: Optional[str] = Field(None, description="Custom X-axis label")
    ylabel: Optional[str] = Field(None, description="Custom Y-axis label")
    palette: Optional[str] = Field("viridis", description="Color palette")


def render_chart_from_spec(spec: ChartSpec, df: pd.DataFrame) -> Dict[str, Any]:
    """Render chart from Pydantic specification and save to outputs/charts/."""
    # Ensure output directory exists
    charts_dir = Path("outputs") / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_id = f"chart_{uuid.uuid4().hex[:8]}"
    file_path = charts_dir / f"{chart_id}.png"

    plt.figure(figsize=(9, 5), dpi=150)
    sns.set_theme(style="whitegrid")

    try:
        if spec.chart_type == "bar":
            if spec.x and spec.y:
                if spec.aggregation:
                    agg_df = getattr(df.groupby(spec.x)[spec.y], spec.aggregation)().reset_index()
                    sns.barplot(data=agg_df, x=spec.x, y=spec.y, hue=spec.x if not spec.hue else spec.hue, palette=spec.palette, legend=False if not spec.hue else True)
                else:
                    sns.barplot(data=df, x=spec.x, y=spec.y, hue=spec.hue if spec.hue else spec.x, palette=spec.palette, legend=False if not spec.hue else True)
            elif spec.x:
                sns.countplot(data=df, x=spec.x, hue=spec.hue if spec.hue else spec.x, palette=spec.palette, legend=False if not spec.hue else True)

        elif spec.chart_type == "countplot":
            sns.countplot(data=df, x=spec.x, hue=spec.hue if spec.hue else spec.x, palette=spec.palette, legend=False if not spec.hue else True)

        elif spec.chart_type == "line":
            sns.lineplot(data=df, x=spec.x, y=spec.y, hue=spec.hue, palette=spec.palette, marker="o")

        elif spec.chart_type == "scatter":
            sns.scatterplot(data=df, x=spec.x, y=spec.y, hue=spec.hue, palette=spec.palette, s=70)

        elif spec.chart_type == "histogram":
            sns.histplot(data=df, x=spec.x, hue=spec.hue, kde=True, palette=spec.palette)

        elif spec.chart_type == "box":
            sns.boxplot(data=df, x=spec.x, y=spec.y, hue=spec.hue, palette=spec.palette)

        elif spec.chart_type == "heatmap":
            num_df = df.select_dtypes(include=["number"])
            corr = num_df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

        elif spec.chart_type == "pie":
            if spec.x:
                val_counts = df[spec.x].value_counts().head(7)
                plt.pie(val_counts.values, labels=val_counts.index, autopct="%1.1f%%", startangle=140)

        # Formatting
        plt.title(spec.title, fontsize=14, fontweight="bold", pad=12)
        if spec.xlabel:
            plt.xlabel(spec.xlabel, fontsize=11)
        elif spec.x:
            plt.xlabel(spec.x, fontsize=11)

        if spec.ylabel:
            plt.ylabel(spec.ylabel, fontsize=11)
        elif spec.y:
            plt.ylabel(spec.y, fontsize=11)

        plt.xticks(rotation=25 if spec.chart_type in ["bar", "countplot", "box"] else 0)
        plt.tight_layout()

        plt.savefig(file_path, format="png", bbox_inches="tight")
        plt.close("all")

        # Generate base64 string for direct frontend embedding
        with open(file_path, "rb") as img_f:
            b64_str = base64.b64encode(img_f.read()).decode("utf-8")

        return {
            "success": True,
            "chart_id": chart_id,
            "chart_path": str(file_path.resolve()),
            "base64": b64_str,
            "title": spec.title,
            "error": None,
        }

    except Exception as e:
        plt.close("all")
        return {
            "success": False,
            "chart_id": None,
            "chart_path": None,
            "base64": None,
            "title": spec.title,
            "error": str(e),
        }
