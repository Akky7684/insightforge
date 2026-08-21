"""Auto-ML Predictive Modeling Engine.

Supports:
- Automatic task detection (Classification vs Regression)
- Automated data preprocessing (imputation, one-hot encoding, scaling)
- Ensemble training with Random Forest & Gradient Boosting
- Evaluation metrics (Accuracy, F1, R², RMSE, MAE)
- Global Feature Importance extraction and ranking
- Publication-grade feature importance visualization
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
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.config import get_settings


def train_predictive_model(
    dataset_path: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    model_type: str = "random_forest",
) -> Dict[str, Any]:
    """Train an Auto-ML model to predict target_column and evaluate performance & feature importance."""
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load dataset with fallback encoding
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="latin1")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns: {df.columns.tolist()}")

    # Drop rows where target is missing
    clean_df = df.dropna(subset=[target_column]).copy()
    if len(clean_df) < 10:
        raise ValueError(f"Insufficient rows ({len(clean_df)}) after dropping missing target values.")

    # 1. Determine Task Type (Classification vs Regression)
    target_series = clean_df[target_column]
    n_unique_targets = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)

    if not is_numeric or n_unique_targets <= 10:
        task_type = "classification"
    else:
        task_type = "regression"

    # 2. Select Features
    if feature_columns:
        valid_features = [c for c in feature_columns if c in clean_df.columns and c != target_column]
    else:
        # Drop ID-like columns, names, tickets, urls
        drop_patterns = ["id", "name", "ticket", "cabin", "url", "description", "uuid", "image"]
        valid_features = [
            c for c in clean_df.columns
            if c != target_column and not any(pat in c.lower() for pat in drop_patterns)
        ]

    if not valid_features:
        raise ValueError("No valid predictive feature columns found in dataset.")

    X_df = clean_df[valid_features].copy()
    y_raw = target_series.copy()

    # 3. Preprocessing: Numeric & Categorical splitting
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    raw_cat_cols = [c for c in X_df.columns if c not in num_cols]

    # Impute numeric missing values
    for col in num_cols:
        X_df[col] = X_df[col].fillna(X_df[col].median() if not X_df[col].dropna().empty else 0.0)

    # Impute & bucket high-cardinality categoricals
    cat_cols = []
    for col in raw_cat_cols:
        series = X_df[col].dropna()
        n_unique = series.nunique()
        if n_unique > 50:
            # Keep top 10 categories, bucket rest into "Other"
            top_10 = series.value_counts().head(10).index.tolist()
            X_df[col] = X_df[col].apply(lambda v: str(v) if v in top_10 else "Other")
        else:
            mode_val = series.mode().iloc[0] if not series.empty else "Unknown"
            X_df[col] = X_df[col].fillna(mode_val).astype(str)
        cat_cols.append(col)

    # Encode categoricals
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_cat = encoder.fit_transform(X_df[cat_cols])
        encoded_cat_feature_names = encoder.get_feature_names_out(cat_cols)
        encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoded_cat_feature_names, index=X_df.index)

        X_processed = pd.concat([X_df[num_cols], encoded_cat_df], axis=1)
    else:
        X_processed = X_df[num_cols].copy()

    feature_names = X_processed.columns.tolist()

    # Encode target if classification
    if task_type == "classification":
        y = y_raw.astype(str)
        classes = sorted(y.unique().tolist())
    else:
        y = y_raw.astype(float)
        classes = []

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=42
    )

    # 4. Train Model (Fast Random Forest / Gradient Boosting)
    if task_type == "classification":
        if model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=60, max_depth=5, random_state=42)
            model_name = "Gradient Boosting Classifier"
        else:
            model = RandomForestClassifier(n_estimators=60, max_depth=10, n_jobs=-1, random_state=42)
            model_name = "Random Forest Classifier"

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = round(float(accuracy_score(y_test, y_pred)), 4)
        f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)

        metrics = {
            "accuracy": acc,
            "accuracy_pct": round(acc * 100, 2),
            "f1_score": f1,
            "total_classes": len(classes),
            "classes": classes,
        }

    else:
        if model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=60, max_depth=5, random_state=42)
            model_name = "Gradient Boosting Regressor"
        else:
            model = RandomForestRegressor(n_estimators=60, max_depth=10, n_jobs=-1, random_state=42)
            model_name = "Random Forest Regressor"

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = round(float(r2_score(y_test, y_pred)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
        mae = round(float(mean_absolute_error(y_test, y_pred)), 2)

        metrics = {
            "r2_score": max(0.0, r2),
            "rmse": rmse,
            "mae": mae,
        }

    # 5. Extract Feature Importance
    raw_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": raw_importances,
    }).sort_values(by="importance", ascending=False)

    top_features = importance_df.head(10).to_dict(orient="records")
    top_features = [
        {"feature": r["feature"], "importance": round(float(r["importance"]), 4), "importance_pct": round(float(r["importance"] * 100), 2)}
        for r in top_features
    ]

    # 6. Generate Feature Importance Visualization
    chart_dir = Path(get_settings().data_dir).parent / "outputs" / "charts"
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = f"predictive_plot_{uuid.uuid4().hex[:8]}.png"
    chart_path = str((chart_dir / chart_filename).resolve())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    top_plot_df = importance_df.head(10).iloc[::-1]  # Reverse for top-down barplot
    sns.barplot(
        data=top_plot_df,
        x="importance",
        y="feature",
        hue="feature",
        palette="crest",
        legend=False,
        ax=ax,
    )

    ax.set_title(f"Top 10 Feature Importances for Predicting: '{target_column}'", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Importance (Gini)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)

    for i, p_bar in enumerate(ax.patches):
        width = p_bar.get_width()
        ax.annotate(
            f"{width:.1%}",
            (width, p_bar.get_y() + p_bar.get_height() / 2.0),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 7. Sample Predictions Preview
    preview_samples = []
    test_sample_indices = min(5, len(y_test))
    for i in range(test_sample_indices):
        preview_samples.append({
            "actual": str(list(y_test)[i]),
            "predicted": str(list(y_pred)[i]) if task_type == "classification" else str(round(float(list(y_pred)[i]), 2)),
        })

    return {
        "dataset_name": p.name,
        "target_column": target_column,
        "task_type": task_type,
        "model_name": model_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_features_used": len(feature_names),
        "metrics": metrics,
        "top_features": top_features,
        "chart_path": chart_path,
        "sample_predictions": preview_samples,
    }
