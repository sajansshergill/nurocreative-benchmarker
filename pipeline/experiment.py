"""
experiment.py
─────────────────────────────────────────────────────────────────────────────
A/B experiment framework for testing whether link posts drive higher
Neural Engagement Score (NES) than self/text posts on Reddit.

Experiment Design:
    Control   : Self posts (text-only, no external URL)
    Treatment : Link posts (external URL — articles, videos, products)

    This mirrors a real product question Meta DS teams answer daily:
    "Does rich-media content drive higher engagement quality than
     plain text posts on a social feed?"

Statistical Approach:
    Primary test   : Welch's t-test (unequal variance assumed)
    Fallback test  : Mann-Whitney U (if normality rejected)
    Effect size    : Cohen's d
    Guardrail      : upvote_ratio must not drop > 2% in treatment

Usage:
    python pipeline/experiment.py --db data/processed/reddit.duckdb
    python pipeline/experiment.py --db data/processed/reddit.duckdb --subreddit technology
    python pipeline/experiment.py --db data/processed/reddit.duckdb --sample 50000
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Experiment Config ────────────────────────────────────────────────────────

ALPHA              = 0.05    # significance level
MIN_SAMPLE_SIZE    = 1_000   # minimum per group for valid test
GUARDRAIL_MAX_DROP = 0.02    # upvote_ratio must not drop more than 2%
MDE                = 0.05    # minimum detectable effect (5% relative lift)
POWER              = 0.80    # desired statistical power


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_experiment_data(
    db_path: Path,
    subreddit: str | None = None,
    sample: int | None = None,
) -> pd.DataFrame:
    """
    Load posts with NES scores from DuckDB for the experiment.

    Classifies posts into control (self/text) and treatment (link) groups
    based on whether the URL is an internal Reddit permalink or external link.

    Args:
        db_path   : Path to DuckDB database.
        subreddit : Optional filter to a single subreddit.
        sample    : Optional row limit per group for faster testing.

    Returns:
        DataFrame with columns:
            post_id, subreddit, score, num_comments, upvote_ratio,
            nes, engagement_tier, group (control|treatment)
    """
    sub_filter = f"AND p.subreddit = '{subreddit}'" if subreddit else ""
    limit_clause = f"LIMIT {sample * 2}" if sample else ""

    query = f"""
        SELECT
            p.post_id,
            p.subreddit,
            p.score,
            p.num_comments,
            p.upvote_ratio,
            p.url,
            n.nes,
            n.engagement_tier,
            -- Classify: self post (text) vs link post (external URL)
            CASE
                WHEN p.url IS NULL
                  OR p.url = ''
                  OR p.url LIKE '%reddit.com%'
                  OR p.url LIKE '%redd.it%'
                THEN 'control'
                ELSE 'treatment'
            END AS "group"
        FROM posts p
        JOIN nes_scores n USING (post_id)
        WHERE n.nes IS NOT NULL
          AND p.upvote_ratio IS NOT NULL
          {sub_filter}
        ORDER BY p.post_id
        {limit_clause}
    """

    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(query).df()
    con.close()

    log.info(
        "Loaded %d posts (%d control / %d treatment)",
        len(df),
        (df["group"] == "control").sum(),
        (df["group"] == "treatment").sum(),
    )
    return df


# ─── Power Analysis ───────────────────────────────────────────────────────────

def power_analysis(n_control: int, n_treatment: int, mean_nes: float) -> dict:
    """
    Check whether sample sizes are sufficient to detect the MDE.

    Uses a simplified normal approximation for t-test power.

    Args:
        n_control   : Number of control group posts.
        n_treatment : Number of treatment group posts.
        mean_nes    : Overall mean NES (used to compute absolute MDE).

    Returns:
        dict with keys: sufficient, min_required, n_control, n_treatment
    """
    # Minimum n per group for 80% power at alpha=0.05, d=0.2 (small effect)
    # Using standard formula: n ≈ 2 * ((z_alpha + z_beta) / d)^2
    z_alpha = 1.96   # two-tailed alpha = 0.05
    z_beta  = 0.84   # power = 0.80
    d_min   = MDE    # minimum detectable effect size (relative)

    # Convert relative MDE to Cohen's d approximation
    absolute_mde = mean_nes * d_min
    assumed_std  = 0.12   # approximate NES std from our data
    cohens_d_mde = absolute_mde / assumed_std

    min_n = int(2 * ((z_alpha + z_beta) / cohens_d_mde) ** 2)

    return {
        "sufficient":   min(n_control, n_treatment) >= min_n,
        "min_required": min_n,
        "n_control":    n_control,
        "n_treatment":  n_treatment,
    }


# ─── Statistical Tests ────────────────────────────────────────────────────────

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation.

    Args:
        group_a : NES scores for group A.
        group_b : NES scores for group B.

    Returns:
        float: Cohen's d (positive = A > B)
    """
    n_a, n_b   = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """Map Cohen's d magnitude to a human-readable label."""
    d = abs(d)
    if d < 0.20:
        return "Negligible"
    elif d < 0.50:
        return "Small"
    elif d < 0.80:
        return "Medium"
    else:
        return "Large"


def run_normality_test(group: np.ndarray, label: str) -> bool:
    """
    Test normality using Shapiro-Wilk on a subsample.

    Returns True if normality is not rejected (p > 0.05).
    """
    sample = group[:5000] if len(group) > 5000 else group
    _, p = stats.shapiro(sample)
    normal = p > 0.05
    log.info(
        "  Normality (%s): shapiro_p=%.4f  %s",
        label, p,
        "normal" if normal else "non-normal"
    )
    return normal


# ─── Guardrail Check ──────────────────────────────────────────────────────────

def check_guardrail(
    control: pd.DataFrame,
    treatment: pd.DataFrame,
) -> dict:
    """
    Check guardrail metric: upvote_ratio must not drop > GUARDRAIL_MAX_DROP.

    In production A/B testing, guardrails prevent shipping changes that
    improve the primary metric but harm user experience elsewhere.

    Args:
        control   : Control group DataFrame.
        treatment : Treatment group DataFrame.

    Returns:
        dict with keys: safe, control_ratio, treatment_ratio, delta
    """
    ctrl_ratio = control["upvote_ratio"].mean()
    trt_ratio  = treatment["upvote_ratio"].mean()
    delta      = trt_ratio - ctrl_ratio
    safe       = delta >= -GUARDRAIL_MAX_DROP

    return {
        "safe":            safe,
        "control_ratio":   round(ctrl_ratio, 4),
        "treatment_ratio": round(trt_ratio, 4),
        "delta":           round(delta, 4),
    }


# ─── Main Experiment Runner ───────────────────────────────────────────────────

def run_experiment(
    db_path: Path,
    subreddit: str | None = None,
    sample: int | None = None,
) -> dict:
    """
    Run the full A/B experiment pipeline.

    Steps:
        1. Load data and split into control / treatment
        2. Power analysis — are sample sizes sufficient?
        3. Normality test — which statistical test to use?
        4. Primary test: Welch's t-test (or Mann-Whitney U fallback)
        5. Effect size: Cohen's d
        6. Guardrail check: upvote_ratio
        7. Recommendation

    Args:
        db_path   : Path to DuckDB database.
        subreddit : Optional subreddit filter.
        sample    : Optional row limit per group.

    Returns:
        dict with full experiment results.
    """
    log.info("=" * 60)
    log.info("A/B EXPERIMENT: Link Posts vs Self Posts")
    log.info("Primary metric : Neural Engagement Score (NES)")
    log.info("Guardrail      : upvote_ratio (max drop: %.0f%%)", GUARDRAIL_MAX_DROP * 100)
    if subreddit:
        log.info("Subreddit filter: r/%s", subreddit)
    log.info("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────
    df = load_experiment_data(db_path, subreddit=subreddit, sample=sample)

    control   = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    n_ctrl = len(control)
    n_trt  = len(treatment)

    if n_ctrl < MIN_SAMPLE_SIZE or n_trt < MIN_SAMPLE_SIZE:
        log.error(
            "Insufficient sample size. Control: %d, Treatment: %d (min: %d)",
            n_ctrl, n_trt, MIN_SAMPLE_SIZE
        )
        raise ValueError("Sample sizes too small for valid experiment.")

    nes_ctrl = control["nes"].values
    nes_trt  = treatment["nes"].values

    # ── Step 2: Power analysis ────────────────────────────────────
    power = power_analysis(n_ctrl, n_trt, df["nes"].mean())
    log.info("─" * 60)
    log.info("POWER ANALYSIS")
    log.info("  Min required per group : %d", power["min_required"])
    log.info("  Control group size     : %d  %s", n_ctrl,
             "✅" if power["sufficient"] else "⚠️  UNDERPOWERED")
    log.info("  Treatment group size   : %d  %s", n_trt,
             "✅" if power["sufficient"] else "⚠️  UNDERPOWERED")

    # ── Step 3: Normality ─────────────────────────────────────────
    log.info("─" * 60)
    log.info("NORMALITY TESTS")
    ctrl_normal = run_normality_test(nes_ctrl, "control")
    trt_normal  = run_normality_test(nes_trt,  "treatment")
    use_parametric = ctrl_normal and trt_normal

    # ── Step 4: Primary statistical test ─────────────────────────
    log.info("─" * 60)
    if use_parametric:
        log.info("PRIMARY TEST: Welch's t-test (normality not rejected)")
        t_stat, p_value = stats.ttest_ind(nes_trt, nes_ctrl, equal_var=False)
        test_name = "Welch's t-test"
    else:
        log.info("PRIMARY TEST: Mann-Whitney U (non-normal distributions)")
        u_stat, p_value = stats.mannwhitneyu(
            nes_trt, nes_ctrl, alternative="two-sided"
        )
        t_stat = u_stat
        test_name = "Mann-Whitney U"

    significant = p_value < ALPHA

    # ── Step 5: Effect size ───────────────────────────────────────
    d = cohens_d(nes_trt, nes_ctrl)
    d_label = interpret_cohens_d(d)

    # ── Step 6: Guardrail ─────────────────────────────────────────
    guardrail = check_guardrail(control, treatment)

    # ── Step 7: Descriptive stats ─────────────────────────────────
    mean_ctrl = float(np.mean(nes_ctrl))
    mean_trt  = float(np.mean(nes_trt))
    abs_lift  = mean_trt - mean_ctrl
    rel_lift  = abs_lift / mean_ctrl * 100 if mean_ctrl > 0 else 0.0

    # ── Winner determination ──────────────────────────────────────
    if not significant:
        winner = "No significant difference"
        recommendation = (
            "Fail to reject H₀. Link posts do not significantly outperform "
            "self posts on NES at α=0.05. Do not change content ranking strategy "
            "based on post type alone."
        )
    elif abs_lift > 0 and guardrail["safe"]:
        winner = "Treatment (Link Posts)"
        recommendation = (
            "Link posts drive significantly higher NES than self posts. "
            "Guardrail metric (upvote_ratio) is safe. "
            "Recommend prioritizing link posts in content ranking."
        )
    elif abs_lift > 0 and not guardrail["safe"]:
        winner = "Treatment (Link Posts) — GUARDRAIL VIOLATED"
        recommendation = (
            "Link posts show higher NES, but upvote_ratio dropped beyond "
            "the guardrail threshold. Do NOT ship. Investigate quality "
            "of link post content before proceeding."
        )
    else:
        winner = "Control (Self Posts)"
        recommendation = (
            "Self posts drive significantly higher NES. "
            "Recommend prioritizing text/self posts in content ranking."
        )

    results = {
        "test_name":       test_name,
        "n_control":       n_ctrl,
        "n_treatment":     n_trt,
        "mean_nes_control":   round(mean_ctrl, 4),
        "mean_nes_treatment": round(mean_trt, 4),
        "absolute_lift":   round(abs_lift, 4),
        "relative_lift_pct": round(rel_lift, 2),
        "statistic":       round(float(t_stat), 4),
        "p_value":         round(float(p_value), 6),
        "significant":     significant,
        "cohens_d":        round(d, 4),
        "effect_size":     d_label,
        "guardrail":       guardrail,
        "winner":          winner,
        "recommendation":  recommendation,
        "power":           power,
    }

    return results


# ─── Report Printer ───────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    """Print a formatted experiment results report."""

    sig_flag = "✅ Significant" if results["significant"] else "❌ Not Significant"
    guard_flag = "✅ Safe" if results["guardrail"]["safe"] else "❌ VIOLATED"

    print("\n")
    print("═" * 60)
    print("        A/B EXPERIMENT RESULTS")
    print("═" * 60)
    print(f"  Test used       : {results['test_name']}")
    print(f"  Sample sizes    : Control={results['n_control']:,}  "
          f"Treatment={results['n_treatment']:,}")
    print()
    print(f"  Control   NES   : {results['mean_nes_control']:.4f}  (self/text posts)")
    print(f"  Treatment NES   : {results['mean_nes_treatment']:.4f}  (link posts)")
    print()
    print(f"  Absolute lift   : {results['absolute_lift']:+.4f}")
    print(f"  Relative lift   : {results['relative_lift_pct']:+.2f}%")
    print()
    print(f"  p-value         : {results['p_value']:.6f}  {sig_flag}")
    print(f"  Cohen's d       : {results['cohens_d']:.4f}  ({results['effect_size']} effect)")
    print()
    print(f"  Guardrail check : upvote_ratio Δ = {results['guardrail']['delta']:+.4f}  {guard_flag}")
    print(f"    Control ratio   : {results['guardrail']['control_ratio']:.4f}")
    print(f"    Treatment ratio : {results['guardrail']['treatment_ratio']:.4f}")
    print()
    print(f"  Winner          : {results['winner']}")
    print()
    print("  Recommendation:")
    print(f"    {results['recommendation']}")
    print("═" * 60)
    print()


# ─── Subreddit Breakdown ──────────────────────────────────────────────────────

def subreddit_breakdown(db_path: Path) -> None:
    """
    Run a per-subreddit NES comparison between link and self posts.
    Shows whether the effect holds across all three subreddits.
    """
    con = duckdb.connect(str(db_path), read_only=True)

    result = con.execute(
        """
        SELECT
            p.subreddit,
            CASE
                WHEN p.url IS NULL
                  OR p.url = ''
                  OR p.url LIKE '%reddit.com%'
                  OR p.url LIKE '%redd.it%'
                THEN 'control'
                ELSE 'treatment'
            END AS grp,
            COUNT(*)           AS n,
            ROUND(AVG(n.nes), 4) AS mean_nes,
            ROUND(STDDEV(n.nes), 4) AS std_nes
        FROM posts p
        JOIN nes_scores n USING (post_id)
        GROUP BY p.subreddit, grp
        ORDER BY p.subreddit, grp
        """
    ).df()

    con.close()

    log.info("─" * 60)
    log.info("NES BY SUBREDDIT AND GROUP")
    log.info("─" * 60)

    for sub in result["subreddit"].unique():
        sub_df = result[result["subreddit"] == sub]
        log.info("  r/%s", sub)
        for _, row in sub_df.iterrows():
            label = "Self/text" if row["grp"] == "control" else "Link posts"
            log.info(
                "    %-12s : n=%6d  mean_NES=%.4f  std=%.4f",
                label, int(row["n"]), row["mean_nes"], row["std_nes"]
            )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run A/B experiment: Link posts vs Self posts on NES."
    )
    parser.add_argument(
        "--db", "-d",
        default="data/processed/reddit.duckdb",
        help="Path to the DuckDB database file.",
    )
    parser.add_argument(
        "--subreddit", "-s",
        default=None,
        help="Optional: filter to a single subreddit (e.g. technology).",
    )
    parser.add_argument(
        "--sample", "-n",
        type=int,
        default=None,
        help="Optional: limit rows loaded (useful for quick testing).",
    )
    parser.add_argument(
        "--breakdown", "-b",
        action="store_true",
        help="Show per-subreddit NES breakdown after main experiment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    db_path = Path(args.db)

    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        log.error("Run ingest.py and metric.py first.")
        raise SystemExit(1)

    # Run experiment
    results = run_experiment(
        db_path,
        subreddit=args.subreddit,
        sample=args.sample,
    )

    # Print formatted report
    print_report(results)

    # Optional subreddit breakdown
    if args.breakdown:
        subreddit_breakdown(db_path)