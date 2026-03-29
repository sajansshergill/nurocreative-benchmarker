-- nes_validation.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- NES metric validation queries.
--
-- These queries answer the question: "Is NES a trustworthy metric?"
-- A Meta DS defining a new metric must validate it before presenting results.
--
-- Validation checks:
--   1. Score distribution (is it reasonable?)
--   2. Correlation with known engagement proxies
--   3. Tier separation quality
--   4. Outlier analysis
--   5. Cross-subreddit consistency
--
-- Run: duckdb data/processed/reddit.duckdb < pipeline/sql/nes_validation.sql
-- ─────────────────────────────────────────────────────────────────────────────


-- ── 1. NES distribution bucketed into deciles ────────────────────────────────

SELECT
    '1. NES DISTRIBUTION (DECILES)'                              AS section,
    ROUND(n.nes, 1)                                              AS nes_bucket,
    COUNT(*)                                                     AS frequency,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)          AS pct,
    ROUND(AVG(p.score), 1)                                       AS avg_score,
    ROUND(AVG(p.num_comments), 1)                                AS avg_comments,
    -- Bar chart proxy
    REPEAT('█', CAST(COUNT(*) * 50 / MAX(COUNT(*)) OVER () AS INTEGER)) AS bar
FROM nes_scores n
JOIN posts p USING (post_id)
GROUP BY ROUND(n.nes, 1)
ORDER BY nes_bucket;


-- ── 2. NES vs score correlation ───────────────────────────────────────────────
-- NES should correlate positively with post score.
-- A weak or negative correlation would suggest the metric is mis-specified.

SELECT
    '2. NES-SCORE CORRELATION'                                   AS section,
    ROUND(CORR(n.nes, p.score), 4)                               AS pearson_r_score,
    ROUND(CORR(n.nes, p.num_comments), 4)                        AS pearson_r_comments,
    ROUND(CORR(n.nes, p.upvote_ratio), 4)                        AS pearson_r_ratio,
    COUNT(*)                                                     AS n
FROM nes_scores n
JOIN posts p USING (post_id);


-- ── 3. NES correlation by score quantile ─────────────────────────────────────
-- Does NES correlate better for high-score or low-score posts?
-- Helps identify if the metric behaves differently across popularity bands.

WITH quantiles AS (
    SELECT
        post_id,
        score,
        NTILE(4) OVER (ORDER BY score) AS score_quartile
    FROM posts
)
SELECT
    '3. NES CORRELATION BY SCORE QUARTILE'                       AS section,
    q.score_quartile,
    MIN(q.score)                                                 AS min_score,
    MAX(q.score)                                                 AS max_score,
    COUNT(*)                                                     AS n,
    ROUND(AVG(n.nes), 4)                                         AS mean_nes,
    ROUND(CORR(n.nes, q.score), 4)                               AS pearson_r
FROM nes_scores n
JOIN quantiles q USING (post_id)
GROUP BY q.score_quartile
ORDER BY q.score_quartile;


-- ── 4. Outlier analysis: extreme NES scores ───────────────────────────────────
-- Inspect posts at the very top and bottom of NES distribution.
-- Outliers reveal edge cases in the metric definition.

SELECT
    '4a. TOP 10 NES POSTS'                                       AS section,
    p.subreddit,
    LEFT(p.title, 65)                                            AS title,
    p.score,
    p.num_comments,
    p.upvote_ratio,
    n.nes,
    n.engagement_tier
FROM nes_scores n
JOIN posts p USING (post_id)
ORDER BY n.nes DESC
LIMIT 10;

SELECT
    '4b. BOTTOM 10 NES POSTS'                                    AS section,
    p.subreddit,
    LEFT(p.title, 65)                                            AS title,
    p.score,
    p.num_comments,
    p.upvote_ratio,
    n.nes,
    n.engagement_tier
FROM nes_scores n
JOIN posts p USING (post_id)
ORDER BY n.nes ASC
LIMIT 10;


-- ── 5. Cross-subreddit NES consistency ───────────────────────────────────────
-- NES should show similar distributions across subreddits if it is
-- a content-agnostic quality signal. Large differences suggest
-- subreddit-specific calibration may be needed.

SELECT
    '5. NES CONSISTENCY ACROSS SUBREDDITS'                       AS section,
    p.subreddit,
    COUNT(*)                                                     AS n,
    ROUND(AVG(n.nes), 4)                                         AS mean_nes,
    ROUND(STDDEV(n.nes), 4)                                      AS std_nes,
    ROUND(MIN(n.nes), 4)                                         AS min_nes,
    ROUND(MAX(n.nes), 4)                                         AS max_nes,
    ROUND(PERCENTILE_CONT(0.25)
        WITHIN GROUP (ORDER BY n.nes), 4)                        AS p25,
    ROUND(PERCENTILE_CONT(0.75)
        WITHIN GROUP (ORDER BY n.nes), 4)                        AS p75,
    -- Coefficient of variation: lower = more consistent metric
    ROUND(STDDEV(n.nes) / AVG(n.nes), 4)                        AS cv
FROM nes_scores n
JOIN posts p USING (post_id)
GROUP BY p.subreddit
ORDER BY mean_nes DESC;


-- ── 6. Metric stability: NES variance over time ───────────────────────────────
-- A stable metric should have consistent variance week over week.
-- Spikes in variance indicate instability in the underlying signals.

SELECT
    '6. WEEKLY METRIC STABILITY'                                 AS section,
    n.week,
    COUNT(*)                                                     AS post_volume,
    ROUND(AVG(n.nes), 4)                                         AS mean_nes,
    ROUND(STDDEV(n.nes), 4)                                      AS std_nes,
    -- CV per week: how noisy is the metric this week?
    ROUND(STDDEV(n.nes) / NULLIF(AVG(n.nes), 0), 4)             AS weekly_cv,
    -- Flag unstable weeks
    CASE
        WHEN STDDEV(n.nes) / NULLIF(AVG(n.nes), 0) > 0.40
        THEN '⚠️  High variance'
        ELSE '✅ Stable'
    END                                                          AS stability_flag
FROM nes_scores n
GROUP BY n.week
HAVING COUNT(*) > 20
ORDER BY n.week;


-- ── 7. Summary validation scorecard ──────────────────────────────────────────

SELECT
    '7. VALIDATION SCORECARD'                                    AS section,
    ROUND(CORR(n.nes, p.score), 4)                               AS nes_score_corr,
    CASE
        WHEN CORR(n.nes, p.score) > 0.30 THEN '✅ PASS'
        WHEN CORR(n.nes, p.score) > 0.10 THEN '⚠️  WEAK'
        ELSE '❌ FAIL'
    END                                                          AS corr_check,
    ROUND(MIN(n.nes), 4)                                         AS min_nes,
    ROUND(MAX(n.nes), 4)                                         AS max_nes,
    CASE
        WHEN MIN(n.nes) >= 0.0
         AND MAX(n.nes) <= 1.0 THEN '✅ PASS'
        ELSE '❌ FAIL'
    END                                                          AS range_check,
    COUNT(DISTINCT n.engagement_tier)                            AS tier_count,
    CASE
        WHEN COUNT(DISTINCT n.engagement_tier) = 3 THEN '✅ PASS'
        ELSE '❌ FAIL'
    END                                                          AS tier_check
FROM nes_scores n
JOIN posts p USING (post_id);