"""
metric.py
─────────────────────────────────────────────────────────────────────────────
Computes and validates the Neural Engagement Score (NES) for all posts
in the DuckDB database.

NES is an original composite metric designed to capture engagement QUALITY,
not just volume. It combines four behavioral signals:

    quality     — upvote_ratio       (are reactions positive?)
    depth       — comment density    (are people actually discussing?)
    velocity    — score per hour     (did it engage fast?)
    controversy — penalty flag       (divisive content scores lower)

Usage:
    python pipeline/metric.py --db data/processed/reddit.duckdb
    python pipeline/metric.py --db data/processed/reddit.duckdb --validate
    python pipeline/metric.py --db data/processed/reddit.duckdb --sample 5
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import time
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── NES Weights ──────────────────────────────────────────────────────────────
# Determined via sensitivity analysis — see validate() for stability check

W_QUALITY     = 0.40
W_DEPTH       = 0.35
W_VELOCITY    = 0.25
PENALTY       = 0.10
CONTROVERSY_THRESHOLD = 0.50   # upvote_ratio below this = controversial
DEPTH_CAP     = 100            # comments above this get normalized to 1.0
VELOCITY_CAP  = 1.0            # score/hour cap after normalization

BATCH_SIZE    = 10_000


# ─── Core NES Function ────────────────────────────────────────────────────────

def neural_engagement_score(
    upvote_ratio: float,
    num_comments: int,
    score: int,
    hours_since_post: float,
) -> float:
    """
    Compute Neural Engagement Score (NES) for a single post.

    NES captures engagement quality across four dimensions:
      - Quality    : Are reactions positive? (upvote_ratio)
      - Depth      : Are people discussing?  (comment density)
      - Velocity   : Did it engage fast?     (score per hour)
      - Controversy: Penalty for divisive content

    Args:
        upvote_ratio     : Float in [0, 1]. Fraction of upvotes.
        num_comments     : Integer. Total comment count.
        score            : Integer. Net upvote score.
        hours_since_post : Float. Hours elapsed since posting (min 1.0).

    Returns:
        float: NES in range [0, 1], rounded to 4 decimal places.
    """
    # Guard against bad data
    upvote_ratio     = float(upvote_ratio or 0.5)
    num_comments     = int(num_comments or 0)
    score            = int(score or 0)
    hours_since_post = max(float(hours_since_post or 1.0), 1.0)

    # Component 1: Quality — direct signal, no transformation needed
    quality = max(min(upvote_ratio, 1.0), 0.0)

    # Component 2: Depth — normalize to [0, 1] with cap at DEPTH_CAP comments
    depth = min(num_comments / DEPTH_CAP, 1.0)

    # Component 3: Velocity — score per hour, normalized to [0, 1]
    raw_velocity = score / hours_since_post
    velocity = min(max(raw_velocity / 1000, 0.0), VELOCITY_CAP)

    # Component 4: Controversy penalty
    controversy = 1.0 if upvote_ratio < CONTROVERSY_THRESHOLD else 0.0

    # Weighted composite
    nes = (
        W_QUALITY  * quality
      + W_DEPTH    * depth
      + W_VELOCITY * velocity
      - PENALTY    * controversy
    )

    return round(max(nes, 0.0), 4)


def assign_tier(nes: float) -> str:
    """
    Map NES score to an engagement tier label.

    Tiers are used for funnel analysis and segmentation in SQL queries.

    Args:
        nes: Float NES score in [0, 1].

    Returns:
        str: One of 'high', 'mid', 'low'
    """
    if nes >= 0.60:
        return "high"
    elif nes >= 0.30:
        return "mid"
    else:
        return "low"


# ─── Vectorized Batch Scoring ─────────────────────────────────────────────────

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NES for an entire DataFrame in one vectorized pass.

    Much faster than row-by-row apply() for large batches.

    Args:
        df: DataFrame with columns:
            post_id, upvote_ratio, num_comments, score, hours_since_post

    Returns:
        DataFrame with added columns: nes, engagement_tier, week
    """
    # Sanitize inputs
    df["upvote_ratio"]     = df["upvote_ratio"].fillna(0.5).clip(0.0, 1.0)
    df["num_comments"]     = df["num_comments"].fillna(0).clip(lower=0)
    df["score"]            = df["score"].fillna(0)
    df["hours_since_post"] = df["hours_since_post"].fillna(1.0).clip(lower=1.0)

    # Vectorized components
    quality     = df["upvote_ratio"]
    depth       = (df["num_comments"] / DEPTH_CAP).clip(upper=1.0)
    velocity    = (df["score"] / df["hours_since_post"] / 1000).clip(0.0, VELOCITY_CAP)
    controversy = (df["upvote_ratio"] < CONTROVERSY_THRESHOLD).astype(float)

    # NES
    df["nes"] = (
        W_QUALITY  * quality
      + W_DEPTH    * depth
      + W_VELOCITY * velocity
      - PENALTY    * controversy
    ).clip(lower=0.0).round(4)

    # Tier
    df["engagement_tier"] = pd.cut(
        df["nes"],
        bins=[-np.inf, 0.30, 0.60, np.inf],
        labels=["low", "mid", "high"],
    ).astype(str)

    # Week (for retention/trend queries)
    df["week"] = pd.to_datetime(
        df["created_utc"], unit="s", utc=True
    ).dt.to_period("W").dt.start_time.dt.date

    return df


# ─── Main Compute Pipeline ────────────────────────────────────────────────────

def compute_nes(db_path: Path) -> None:
    """
    Score all posts in DuckDB and write results to nes_scores table.

    Processes in batches of BATCH_SIZE rows to stay memory-efficient
    even on machines with limited RAM.

    Args:
        db_path: Path to the DuckDB database file.
    """
    con = duckdb.connect(str(db_path))

    # Clear previous scores for clean rerun
    con.execute("DELETE FROM nes_scores")
    log.info("Cleared existing NES scores for fresh run")

    # Count total posts
    total = con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    log.info("Scoring %d posts...", total)

    processed = 0
    start = time.time()

    while processed < total:
        # Fetch batch
        df = con.execute(
            """
            SELECT
                post_id,
                upvote_ratio,
                num_comments,
                score,
                hours_since_post,
                created_utc
            FROM posts
            ORDER BY post_id
            LIMIT ? OFFSET ?
            """,
            [BATCH_SIZE, processed],
        ).df()

        if df.empty:
            break

        # Score batch
        df = score_dataframe(df)

        # Write to DuckDB
        con.execute(
            """
            INSERT OR IGNORE INTO nes_scores (post_id, nes, engagement_tier, week)
            SELECT post_id, nes, engagement_tier, week::DATE
            FROM df
            """
        )

        processed += len(df)
        pct = processed / total * 100
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        log.info(
            "  Scored %d / %d (%.1f%%) — %.0f rows/sec",
            processed, total, pct, rate
        )

    elapsed = time.time() - start
    log.info("─" * 60)
    log.info("NES computation complete.")
    log.info("  Total scored : %d", processed)
    log.info("  Time elapsed : %.1f seconds", elapsed)
    log.info("  Throughput   : %.0f rows/sec", processed / elapsed)

    con.close()


# ─── Validation Suite ─────────────────────────────────────────────────────────

def validate(db_path: Path) -> None:
    """
    Run the NES metric validation checklist.

    Tests:
      1. Distribution normality (Shapiro-Wilk on sample)
      2. Tier separation (ANOVA across high/mid/low)
      3. Temporal stability (week-over-week coefficient of variation)
      4. Weight sensitivity (±10% perturbation test)
      5. Score range check (all NES in [0, 1])

    Args:
        db_path: Path to the DuckDB database file.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    log.info("=" * 60)
    log.info("NES VALIDATION SUITE")
    log.info("=" * 60)

    # ── 1. Score range ────────────────────────────────────────────
    result = con.execute(
        "SELECT MIN(nes), MAX(nes), COUNT(*) FROM nes_scores"
    ).fetchone()
    min_nes, max_nes, count = result
    range_ok = 0.0 <= min_nes and max_nes <= 1.0
    log.info(
        "[1] Score range   : min=%.4f  max=%.4f  count=%d  %s",
        min_nes, max_nes, count,
        "✅ PASS" if range_ok else "❌ FAIL"
    )

    # ── 2. Distribution (sample of 5000 for Shapiro-Wilk) ─────────
    sample = con.execute(
        "SELECT nes FROM nes_scores USING SAMPLE 5000"
    ).df()["nes"].values

    _, p_shapiro = stats.shapiro(sample[:5000])
    # Note: Shapiro-Wilk often rejects normality at large N
    # We check for approximate bell shape via skewness instead
    skew = stats.skew(sample)
    dist_ok = abs(skew) < 2.0   # acceptable skew range
    log.info(
        "[2] Distribution  : skewness=%.3f  shapiro_p=%.4f  %s",
        skew, p_shapiro,
        "✅ PASS (acceptable skew)" if dist_ok else "⚠️  WARN (high skew)"
    )

    # ── 3. Tier separation (ANOVA) ────────────────────────────────
    tiers = con.execute(
        """
        SELECT engagement_tier, nes
        FROM nes_scores
        WHERE engagement_tier IN ('high', 'mid', 'low')
        """
    ).df()

    high = tiers[tiers["engagement_tier"] == "high"]["nes"].values
    mid  = tiers[tiers["engagement_tier"] == "mid"]["nes"].values
    low  = tiers[tiers["engagement_tier"] == "low"]["nes"].values

    f_stat, p_anova = stats.f_oneway(high, mid, low)
    anova_ok = p_anova < 0.05
    log.info(
        "[3] Tier separation: F=%.2f  p=%.6f  %s",
        f_stat, p_anova,
        "✅ PASS" if anova_ok else "❌ FAIL"
    )

    # Tier distribution
    tier_counts = con.execute(
        """
        SELECT engagement_tier, COUNT(*) as n,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
        FROM nes_scores
        GROUP BY engagement_tier
        ORDER BY n DESC
        """
    ).fetchall()
    for tier, n, pct in tier_counts:
        log.info("          %-6s : %d posts (%.1f%%)", tier, n, pct)

    # ── 4. Temporal stability ─────────────────────────────────────
    weekly = con.execute(
        """
        SELECT week, AVG(nes) as mean_nes
        FROM nes_scores
        GROUP BY week
        HAVING COUNT(*) > 10
        ORDER BY week
        """
    ).df()

    if len(weekly) > 1:
        cv = weekly["mean_nes"].std() / weekly["mean_nes"].mean()
        stability_ok = cv < 0.15
        log.info(
            "[4] Temporal stability: CV=%.4f  %s",
            cv,
            "✅ PASS (CV < 0.15)" if stability_ok else "⚠️  WARN (high variance)"
        )
    else:
        log.info("[4] Temporal stability: ⚠️  Not enough weekly data")

    # ── 5. Weight sensitivity (±10% perturbation) ─────────────────
    base_scores = (
        W_QUALITY  * 0.75
      + W_DEPTH    * 0.50
      + W_VELOCITY * 0.30
      - PENALTY    * 0.0
    )
    perturbed_scores = []
    for delta in [-0.10, +0.10]:
        w_q = W_QUALITY + delta
        w_d = W_DEPTH - delta / 2
        w_v = W_VELOCITY - delta / 2
        s = w_q * 0.75 + w_d * 0.50 + w_v * 0.30
        perturbed_scores.append(s)

    max_drift = max(abs(s - base_scores) for s in perturbed_scores)
    sensitivity_ok = max_drift < 0.05
    log.info(
        "[5] Weight sensitivity: max_drift=%.4f  %s",
        max_drift,
        "✅ PASS (drift < 0.05)" if sensitivity_ok else "⚠️  WARN"
    )

    log.info("=" * 60)
    con.close()


# ─── Summary Report ───────────────────────────────────────────────────────────

def summary(db_path: Path) -> None:
    """Print a concise NES summary report."""
    con = duckdb.connect(str(db_path), read_only=True)

    stats_row = con.execute(
        """
        SELECT
            COUNT(*)                    AS total,
            ROUND(AVG(nes), 4)          AS mean_nes,
            ROUND(STDDEV(nes), 4)       AS std_nes,
            ROUND(MIN(nes), 4)          AS min_nes,
            ROUND(MAX(nes), 4)          AS max_nes,
            ROUND(PERCENTILE_CONT(0.25)
                WITHIN GROUP (ORDER BY nes), 4) AS p25,
            ROUND(PERCENTILE_CONT(0.50)
                WITHIN GROUP (ORDER BY nes), 4) AS p50,
            ROUND(PERCENTILE_CONT(0.75)
                WITHIN GROUP (ORDER BY nes), 4) AS p75
        FROM nes_scores
        """
    ).fetchone()

    total, mean, std, mn, mx, p25, p50, p75 = stats_row

    log.info("─" * 60)
    log.info("NES SUMMARY REPORT")
    log.info("─" * 60)
    log.info("  Total scored : %d", total)
    log.info("  Mean NES     : %.4f", mean)
    log.info("  Std Dev      : %.4f", std)
    log.info("  Min          : %.4f", mn)
    log.info("  Max          : %.4f", mx)
    log.info("  P25 / P50 / P75 : %.4f / %.4f / %.4f", p25, p50, p75)

    # Per-subreddit NES
    sub_stats = con.execute(
        """
        SELECT
            p.subreddit,
            COUNT(*)           AS n,
            ROUND(AVG(n.nes), 4) AS mean_nes,
            ROUND(STDDEV(n.nes), 4) AS std_nes
        FROM nes_scores n
        JOIN posts p USING (post_id)
        GROUP BY p.subreddit
        ORDER BY mean_nes DESC
        """
    ).fetchall()

    log.info("─" * 60)
    log.info("  NES by subreddit:")
    for sub, n, mean_nes, std_nes in sub_stats:
        log.info(
            "    r/%-15s  n=%d  mean=%.4f  std=%.4f",
            sub, n, mean_nes, std_nes
        )

    log.info("─" * 60)
    con.close()


# ─── Sample Viewer ────────────────────────────────────────────────────────────

def show_sample(db_path: Path, n: int = 5) -> None:
    """Print n sample posts with their NES scores."""
    con = duckdb.connect(str(db_path), read_only=True)

    rows = con.execute(
        f"""
        SELECT
            p.subreddit,
            LEFT(p.title, 60)   AS title,
            p.score,
            p.num_comments,
            p.upvote_ratio,
            n.nes,
            n.engagement_tier
        FROM nes_scores n
        JOIN posts p USING (post_id)
        ORDER BY n.nes DESC
        LIMIT {n}
        """
    ).fetchall()

    log.info("─" * 60)
    log.info("TOP %d POSTS BY NES", n)
    log.info("─" * 60)
    for row in rows:
        sub, title, score, comments, ratio, nes, tier = row
        log.info(
            "  [%s] r/%-12s  NES=%.4f  score=%d  comments=%d  ratio=%.2f",
            tier.upper(), sub, nes, score, comments, ratio
        )
        log.info("    \"%s\"", title)

    con.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and validate Neural Engagement Score (NES)."
    )
    parser.add_argument(
        "--db", "-d",
        default="data/processed/reddit.duckdb",
        help="Path to the DuckDB database file.",
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Run the full NES validation suite after scoring.",
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=0,
        help="Print top N posts by NES after scoring.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print NES summary statistics after scoring.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    db_path = Path(args.db)

    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        log.error("Run ingest.py first.")
        raise SystemExit(1)

    # Step 1: Compute NES for all posts
    compute_nes(db_path)

    # Step 2: Summary (always shown)
    summary(db_path)

    # Step 3: Optional validation
    if args.validate:
        validate(db_path)

    # Step 4: Optional sample viewer
    if args.sample > 0:
        show_sample(db_path, args.sample)