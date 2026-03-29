-- retention_weekly.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Weekly NES trend and retention-style analysis.
--
-- "Retention" here means: does engagement quality hold up over time?
-- We track mean NES per week and look for decay, growth, or seasonality.
--
-- This mirrors how Meta DS teams track product health metrics over time —
-- not just a snapshot, but a trend that informs investment decisions.
--
-- Run: duckdb data/processed/reddit.duckdb < pipeline/sql/retention_weekly.sql
-- ─────────────────────────────────────────────────────────────────────────────


-- ── 1. Weekly NES trend with week-over-week delta ────────────────────────────

SELECT
    '1. WEEKLY NES TREND'                                        AS section,
    n.week,
    COUNT(*)                                                     AS post_volume,
    ROUND(AVG(n.nes), 4)                                         AS mean_nes,
    ROUND(PERCENTILE_CONT(0.5)
        WITHIN GROUP (ORDER BY n.nes), 4)                        AS median_nes,
    ROUND(STDDEV(n.nes), 4)                                      AS std_nes,
    ROUND(
        AVG(n.nes)
        - LAG(AVG(n.nes)) OVER (ORDER BY n.week),
    4)                                                           AS wow_delta,
    -- Rolling 4-week average for smoothing
    ROUND(AVG(AVG(n.nes)) OVER (
        ORDER BY n.week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ), 4)                                                        AS rolling_4w_avg
FROM nes_scores n
GROUP BY n.week
HAVING COUNT(*) > 10
ORDER BY n.week;


-- ── 2. Weekly trend by subreddit ──────────────────────────────────────────────

SELECT
    '2. WEEKLY TREND BY SUBREDDIT'                               AS section,
    n.week,
    p.subreddit,
    COUNT(*)                                                     AS post_volume,
    ROUND(AVG(n.nes), 4)                                         AS mean_nes,
    ROUND(
        AVG(n.nes)
        - LAG(AVG(n.nes)) OVER (
            PARTITION BY p.subreddit
            ORDER BY n.week
          ),
    4)                                                           AS wow_delta
FROM nes_scores n
JOIN posts p USING (post_id)
GROUP BY n.week, p.subreddit
HAVING COUNT(*) > 5
ORDER BY p.subreddit, n.week;


-- ── 3. Monthly aggregation: higher-level trend view ───────────────────────────

SELECT
    '3. MONTHLY NES TREND'                                       AS section,
    DATE_TRUNC('month', n.week::TIMESTAMP)::DATE                 AS month,
    COUNT(*)                                                     AS post_volume,
    ROUND(AVG(n.nes), 4)                                         AS mean_nes,
    ROUND(STDDEV(n.nes), 4)                                      AS std_nes,
    -- Month-over-month delta
    ROUND(
        AVG(n.nes)
        - LAG(AVG(n.nes)) OVER (
            ORDER BY DATE_TRUNC('month', n.week::TIMESTAMP)
          ),
    4)                                                           AS mom_delta,
    -- High-tier post rate per month
    ROUND(
        SUM(CASE WHEN n.engagement_tier = 'high' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*),
    2)                                                           AS high_tier_pct
FROM nes_scores n
GROUP BY DATE_TRUNC('month', n.week::TIMESTAMP)
HAVING COUNT(*) > 50
ORDER BY month;


-- ── 4. Engagement decay: NES vs post age ─────────────────────────────────────
-- Does engagement quality decay as posts age?
-- Buckets posts by age in days and computes mean NES per bucket.

SELECT
    '4. ENGAGEMENT DECAY BY POST AGE'                            AS section,
    CASE
        WHEN p.hours_since_post < 24        THEN '< 1 day'
        WHEN p.hours_since_post < 24 * 7   THEN '1-7 days'
        WHEN p.hours_since_post < 24 * 30  THEN '1-4 weeks'
        WHEN p.hours_since_post < 24 * 90  THEN '1-3 months'
        WHEN p.hours_since_post < 24 * 365 THEN '3-12 months'
        ELSE '> 1 year'
    END                                                          AS age_bucket,
    COUNT(*)                                                     AS post_count,
    ROUND(AVG(n.nes), 4)                                         AS mean_nes,
    ROUND(AVG(p.score), 1)                                       AS avg_score,
    ROUND(AVG(p.num_comments), 1)                                AS avg_comments
FROM nes_scores n
JOIN posts p USING (post_id)
GROUP BY age_bucket
ORDER BY
    CASE age_bucket
        WHEN '< 1 day'      THEN 1
        WHEN '1-7 days'     THEN 2
        WHEN '1-4 weeks'    THEN 3
        WHEN '1-3 months'   THEN 4
        WHEN '3-12 months'  THEN 5
        ELSE 6
    END;


-- ── 5. Best and worst weeks by NES ───────────────────────────────────────────

WITH weekly AS (
    SELECT
        n.week,
        COUNT(*)         AS post_volume,
        ROUND(AVG(n.nes), 4) AS mean_nes
    FROM nes_scores n
    GROUP BY n.week
    HAVING COUNT(*) > 50
),
ranked AS (
    SELECT
        week,
        post_volume,
        mean_nes,
        RANK() OVER (ORDER BY mean_nes DESC) AS rank_high,
        RANK() OVER (ORDER BY mean_nes ASC)  AS rank_low
    FROM weekly
)
SELECT
    '5. BEST AND WORST WEEKS'                                    AS section,
    week,
    post_volume,
    mean_nes,
    CASE
        WHEN rank_high <= 5 THEN 'Top 5'
        WHEN rank_low  <= 5 THEN 'Bottom 5'
    END                                                          AS ranking
FROM ranked
WHERE rank_high <= 5 OR rank_low <= 5
ORDER BY mean_nes DESC;