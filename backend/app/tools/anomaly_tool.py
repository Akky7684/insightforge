"""Anomaly Detection Engine using scikit-learn.

Supports:
- Isolation Forest (Multivariate Global Anomalies)
- Local Outlier Factor (Density-based Local Anomalies)
- Robust Z-Score / Median Absolute Deviation (Univariate Extremes)
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
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from backend.app.config import get_settings


def detect_anomalies(
    dataset_path: str,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Detect anomalies in a dataset using specified algorithm.
    Returns flagged rows, anomaly counts, and a visual 2D scatter plot path.
    """
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load data
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="latin1")

    # Select numerical columns
    if features:
        num_cols = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Drop ID-like columns automatically if not explicitly requested
    if not features:
        num_cols = [c for c in num_cols if not c.lower().endswith("id") and c.lower() != "id"]

    if len(num_cols) == 0:
        raise ValueError("No numerical columns available for anomaly detection.")

    # Prepare data (impute NaNs with median)
    X_raw = df[num_cols].copy()
    X = X_raw.fillna(X_raw.median())

    # Scale data for distance-based / variance-based models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    anomalies = np.zeros(len(df), dtype=bool)

    # 1. Apply Anomaly Algorithm
    if method == "isolation_forest":
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X_scaled)
        anomalies = preds == -1

    elif method == "lof":
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        preds = lof.fit_predict(X_scaled)
        anomalies = preds == -1

    elif method == "zscore":
        # Robust Z-Score using Median Absolute Deviation (MAD)
        # Z = 0.6745 * (X - Median) / MAD
        median = X.median()
        mad = np.abs(X - median).median()
        # Avoid division by zero
        mad = mad.replace(0, 1e-6)
        robust_z = 0.6745 * (X - median) / mad
        
        # Determine threshold based on contamination (assume max Z across columns)
        max_z_per_row = robust_z.abs().max(axis=1)
        threshold = np.percentile(max_z_per_row, 100 * (1 - contamination))
        anomalies = (max_z_per_row >= threshold).to_numpy()

    else:
        raise ValueError(f"Unknown anomaly detection method: {method}")

    total_anomalies = int(anomalies.sum())
    
    # 2. Extract Anomalous Rows
    # Attach original index for reference
    df_result = df.copy()
    df_result["is_anomaly"] = anomalies
    
    anomalous_df = df_result[df_result["is_anomaly"]].copy()
    # Drop the boolean flag for final JSON payload to save space
    anomalous_df = anomalous_df.drop(columns=["is_anomaly"])
    
    # Take top N if there are too many to fit in prompt context
    max_return_rows = 50
    sampled_anomalies = anomalous_df.head(max_return_rows).to_dict(orient="records")

    # 3. Generate 2D Visual Scatter Plot (PCA if > 2 dims)
    chart_dir = Path(get_settings().data_dir).parent / "outputs" / "charts"
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = f"anomaly_plot_{method}_{uuid.uuid4().hex[:8]}.png"
    chart_path = str((chart_dir / chart_filename).resolve())

    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(num_cols) >= 2:
        if len(num_cols) > 2:
            pca = PCA(n_components=2, random_state=42)
            plot_data = pca.fit_transform(X_scaled)
            x_label = f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} var)"
            y_label = f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} var)"
        else:
            plot_data = X_scaled
            x_label = num_cols[0] + " (Standardized)"
            y_label = num_cols[1] + " (Standardized)"

        scatter = ax.scatter(
            plot_data[:, 0], 
            plot_data[:, 1], 
            c=np.where(anomalies, "#ef4444", "#3b82f6"), # Red for anomalies, Blue for normal
            alpha=np.where(anomalies, 0.9, 0.3),
            s=np.where(anomalies, 50, 20),
            edgecolors='w',
            linewidths=0.5
        )
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f'Normal ({len(df)-total_anomalies:,})', markerfacecolor='#3b82f6', markersize=8),
            Line2D([0], [0], marker='o', color='w', label=f'Anomaly ({total_anomalies:,})', markerfacecolor='#ef4444', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc="best", title=f"Method: {method.upper()}")
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Anomaly Detection Scatter Plot ({contamination:.1%} contamination)", fontweight="bold")
    else:
        # 1D plot
        ax.scatter(
            X_scaled[:, 0], 
            np.zeros_like(X_scaled[:, 0]), 
            c=np.where(anomalies, "#ef4444", "#3b82f6"),
            alpha=0.5
        )
        ax.set_xlabel(num_cols[0] + " (Standardized)")
        ax.set_yticks([])
        ax.set_title(f"1D Anomaly Detection ({contamination:.1%} contamination)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "dataset_name": p.name,
        "method": method,
        "contamination": contamination,
        "features_analyzed": num_cols,
        "total_records": len(df),
        "total_anomalies": total_anomalies,
        "anomaly_percentage": round((total_anomalies / len(df)) * 100, 2),
        "anomalous_rows": sampled_anomalies,
        "chart_path": chart_path
    }
