"""Statistical Hypothesis Testing Tool — automated assumption checking and effect size analysis.

Supports:
- Independent Samples T-Test (with Cohen's d effect size & Shapiro normality check)
- Chi-Square Test of Independence (with Cramér's V effect size)
- One-Way ANOVA (with Eta-squared effect size)
- Mann-Whitney U Test (non-parametric two-group comparison)
- Pearson & Spearman Correlation Tests

Returns structured Pydantic results with plain-English interpretations.
"""

from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import scipy.stats as stats


class StatsTestRequest(BaseModel):
    """Specification for running an automated hypothesis test."""

    test_type: Literal[
        "t_test_ind",
        "chi2_contingency",
        "anova",
        "mann_whitney",
        "pearson_corr",
        "spearman_corr",
    ] = Field(..., description="Hypothesis test type")
    var1: str = Field(..., description="Primary variable / column name")
    var2: Optional[str] = Field(None, description="Secondary variable / column name")
    group_col: Optional[str] = Field(None, description="Grouping column for two-sample or ANOVA comparisons")
    alpha: float = Field(default=0.05, description="Significance threshold (alpha level)")


class StatsTestResult(BaseModel):
    """Structured hypothesis test result with effect sizes and interpretations."""

    test_name: str
    statistic: float
    p_value: float
    alpha: float
    is_significant: bool
    effect_size: Optional[Dict[str, Any]] = None
    sample_sizes: Dict[str, int]
    assumptions_summary: Dict[str, Any]
    interpretation: str


def run_stats_test(df: pd.DataFrame, req: StatsTestRequest) -> StatsTestResult:
    """Execute the requested hypothesis test on DataFrame with assumption checks and effect size."""
    alpha = req.alpha

    # 1. Independent Two-Sample T-Test
    if req.test_type == "t_test_ind":
        if not req.group_col or req.group_col not in df.columns:
            raise ValueError(f"group_col '{req.group_col}' must be provided for independent t-test.")
        if req.var1 not in df.columns:
            raise ValueError(f"var1 '{req.var1}' not found in dataset.")

        groups = df[req.group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"group_col must contain exactly 2 unique values for t-test. Found {len(groups)}: {groups[:5]}")

        g1 = df[df[req.group_col] == groups[0]][req.var1].dropna().values
        g2 = df[df[req.group_col] == groups[1]][req.var1].dropna().values

        stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

        # Cohen's d effect size
        n1, n2 = len(g1), len(g2)
        s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
        cohens_d = round(float((np.mean(g1) - np.mean(g2)) / pooled_sd), 3) if pooled_sd > 0 else 0.0

        # Normality checks (Shapiro-Wilk on sample up to 500)
        norm_g1 = stats.shapiro(g1[:500])[1] if len(g1) >= 3 else None
        norm_g2 = stats.shapiro(g2[:500])[1] if len(g2) >= 3 else None

        is_sig = bool(p_val < alpha)
        interp = (
            f"Statistically significant difference detected (t={stat:.2f}, p={p_val:.4f} < {alpha}) between "
            f"'{groups[0]}' (mean={np.mean(g1):.2f}) and '{groups[1]}' (mean={np.mean(g2):.2f}) with Cohen's d={cohens_d}."
            if is_sig
            else f"No statistically significant difference (t={stat:.2f}, p={p_val:.4f} >= {alpha}) between '{groups[0]}' and '{groups[1]}'."
        )

        return StatsTestResult(
            test_name="Welch's Independent Two-Sample T-Test",
            statistic=round(float(stat), 4),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            is_significant=is_sig,
            effect_size={"metric": "Cohen's d", "value": cohens_d},
            sample_sizes={str(groups[0]): len(g1), str(groups[1]): len(g2)},
            assumptions_summary={
                "normality_p_group1": round(float(norm_g1), 4) if norm_g1 else None,
                "normality_p_group2": round(float(norm_g2), 4) if norm_g2 else None,
            },
            interpretation=interp,
        )

    # 2. Chi-Square Test of Independence
    elif req.test_type == "chi2_contingency":
        if not req.var2 or req.var2 not in df.columns:
            raise ValueError(f"var2 '{req.var2}' must be provided for Chi-Square test.")
        if req.var1 not in df.columns:
            raise ValueError(f"var1 '{req.var1}' not found in dataset.")

        ct = pd.crosstab(df[req.var1], df[req.var2])
        chi2, p_val, dof, _ = stats.chi2_contingency(ct)

        # Cramér's V effect size
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        cramers_v = round(float(np.sqrt((chi2 / n) / min_dim)), 3) if min_dim > 0 and n > 0 else 0.0

        is_sig = bool(p_val < alpha)
        interp = (
            f"Statistically significant association found between '{req.var1}' and '{req.var2}' "
            f"(Chi2={chi2:.2f}, df={dof}, p={p_val:.4e} < {alpha}, Cramér's V={cramers_v})."
            if is_sig
            else f"No significant association between '{req.var1}' and '{req.var2}' (Chi2={chi2:.2f}, p={p_val:.4f} >= {alpha})."
        )

        return StatsTestResult(
            test_name="Chi-Square Test of Independence",
            statistic=round(float(chi2), 4),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            is_significant=is_sig,
            effect_size={"metric": "Cramér's V", "value": cramers_v},
            sample_sizes={"total_observations": int(n)},
            assumptions_summary={"degrees_of_freedom": int(dof)},
            interpretation=interp,
        )

    # 3. One-Way ANOVA
    elif req.test_type == "anova":
        if not req.group_col or req.group_col not in df.columns:
            raise ValueError(f"group_col '{req.group_col}' must be provided for ANOVA.")

        groups = df[req.group_col].dropna().unique()
        group_samples = [df[df[req.group_col] == g][req.var1].dropna().values for g in groups]
        group_samples = [g for g in group_samples if len(g) > 0]

        stat, p_val = stats.f_oneway(*group_samples)

        # Eta-squared (SS_between / SS_total)
        all_vals = np.concatenate(group_samples)
        grand_mean = np.mean(all_vals)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_samples)
        eta_sq = round(float(ss_between / ss_total), 3) if ss_total > 0 else 0.0

        is_sig = bool(p_val < alpha)
        interp = (
            f"Statistically significant difference across '{req.group_col}' groups on '{req.var1}' "
            f"(F={stat:.2f}, p={p_val:.4e} < {alpha}, Eta-squared={eta_sq})."
            if is_sig
            else f"No significant group differences across '{req.group_col}' (F={stat:.2f}, p={p_val:.4f} >= {alpha})."
        )

        return StatsTestResult(
            test_name="One-Way ANOVA",
            statistic=round(float(stat), 4),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            is_significant=is_sig,
            effect_size={"metric": "Eta-squared", "value": eta_sq},
            sample_sizes={str(g): len(s) for g, s in zip(groups, group_samples)},
            assumptions_summary={"num_groups": len(group_samples)},
            interpretation=interp,
        )

    # 4. Mann-Whitney U Test (Non-parametric)
    elif req.test_type == "mann_whitney":
        if not req.group_col or req.group_col not in df.columns:
            raise ValueError("group_col required for Mann-Whitney test.")
        groups = df[req.group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError("group_col must contain exactly 2 unique groups.")

        g1 = df[df[req.group_col] == groups[0]][req.var1].dropna().values
        g2 = df[df[req.group_col] == groups[1]][req.var1].dropna().values

        stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        # Rank-biserial correlation effect size = 1 - (2*U)/(n1*n2)
        n1, n2 = len(g1), len(g2)
        r_biserial = round(float(1.0 - (2.0 * stat) / (n1 * n2)), 3) if (n1 * n2) > 0 else 0.0

        is_sig = bool(p_val < alpha)
        interp = (
            f"Statistically significant difference in distributions between '{groups[0]}' and '{groups[1]}' "
            f"(Mann-Whitney U={stat:.1f}, p={p_val:.4e} < {alpha}, Rank-biserial r={r_biserial})."
            if is_sig
            else f"No significant difference in distributions (U={stat:.1f}, p={p_val:.4f} >= {alpha})."
        )

        return StatsTestResult(
            test_name="Mann-Whitney U Test",
            statistic=round(float(stat), 4),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            is_significant=is_sig,
            effect_size={"metric": "Rank-biserial r", "value": r_biserial},
            sample_sizes={str(groups[0]): len(g1), str(groups[1]): len(g2)},
            assumptions_summary={"non_parametric": True},
            interpretation=interp,
        )

    # 5. Pearson / Spearman Correlation
    elif req.test_type in ["pearson_corr", "spearman_corr"]:
        if not req.var2 or req.var2 not in df.columns:
            raise ValueError("var2 must be provided for correlation test.")

        valid_df = df[[req.var1, req.var2]].dropna()
        x, y = valid_df[req.var1].values, valid_df[req.var2].values

        if req.test_type == "pearson_corr":
            stat, p_val = stats.pearsonr(x, y)
            name = "Pearson Correlation"
        else:
            stat, p_val = stats.spearmanr(x, y)
            name = "Spearman Rank Correlation"

        is_sig = bool(p_val < alpha)
        interp = (
            f"Statistically significant correlation between '{req.var1}' and '{req.var2}' "
            f"(r={stat:.3f}, p={p_val:.4e} < {alpha})."
            if is_sig
            else f"No significant linear correlation (r={stat:.3f}, p={p_val:.4f} >= {alpha})."
        )

        return StatsTestResult(
            test_name=name,
            statistic=round(float(stat), 4),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            is_significant=is_sig,
            effect_size={"metric": "Correlation Coefficient (r)", "value": round(float(stat), 3)},
            sample_sizes={"valid_pairs": len(valid_df)},
            assumptions_summary={},
            interpretation=interp,
        )

    else:
        raise ValueError(f"Unsupported test type: {req.test_type}")
