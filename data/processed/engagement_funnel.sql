-- engagement_funnel.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Engagement funnel analysis: how posts distribute across NES tiers
-- and what behavioral signals characterize each tier.
--
-- Think of this as a classic product funnel:
--   All posts → Low engagement → Mid engagement → High engagement
--
-- Run: duckdb data/processed/reddit.duckdb < pipeline/sql/engagement_funnel.sql
-- ─────────────────────────────────────────────────────────────────────────────


-- ── 1. Overall funnel: tier distribution across all posts ─────────────────────

SELECT
    '1. OVERALL FUNNEL'                                    AS section,
    n.engagement_tier,
    COUNT(*)                                               AS post_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)    AS pct_of_total,
    ROUND(AVG(n.nes), 4)                                   AS mean_nes,
    ROUND(AVG(p.score), 1)                                 AS avg_score,
    ROUND(AVG(p.num_comments), 1)                          AS avg_comments,
    ROUND(AVG(p.upvote_ratio), 4)                          AS avg_upvote_ratio
FROM nes_scores n
JOIN posts p USING (post_id)
GROUP BY n.engagement_tier
ORDER BY mean_nes DESC;


-- ── 2. Funnel by subreddit: where does each community sit? ────────────────────

SELECT
    '2. FUNNEL BY SUBREDDIT'                               AS section,
    p.subreddit,
    n.engagement_tier,
    COUNT(*)                                               AS post_count,
    ROUND(COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER (PARTITION BY p.subreddit), 2) AS pct_within_sub,
    ROUND(AVG(n.nes), 4)                                   AS mean_nes,
    ROUND(AVG(p.score), 1)                                 AS avg_score,
    ROUND(AVG(p.num_comments), 1)                          AS avg_comments
FROM nes_scores n
JOIN posts p USING (post_id)
GROUP BY p.subreddit, n.engagement_tier
ORDER BY p.subreddit, mean_nes DESC;


-- ── 3. Funnel by post type: link vs self posts ────────────────────────────────

SELECT
    '3. FUNNEL BY POST TYPE'                               AS section,
    CASE
        WHEN p.url IS NULL
          OR p.url = ''
          OR p.url LIKE '%reddit.com%'
          OR p.url LIKE '%redd.it%'
        THEN 'self_post'
        ELSE 'link_post'
    END                                                    AS post_type,
    n.engagement_tier,
    COUNT(*)                                               AS post_count,
    ROUND(COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER (
            PARTITION BY CASE
                WHEN p.url IS NULL
                  OR p.url = ''
                  OR p.url LIKE '%reddit.com%'
                  OR p.url LIKE '%redd.it%'
                THEN 'self_post'
                ELSE 'link_post'
            END
          ), 2)                                            AS pct_within_type,
    ROUND(AVG(n.nes), 4)                                   AS mean_nes,
    ROUND(AVG(p.score), 1)                                 AS avg_score
FROM nes_scores n
JOIN posts p USING (post_id)
GROUP BY post_type, n.engagement_tier
ORDER BY post_type, mean_nes DESC;


-- ── 4. Top converting posts: low score but high NES ───────────────────────────
-- These are hidden gems — posts that engaged deeply despite low visibility.
-- A content ranker should surface these.

SELECT
    '4. HIDDEN GEMS (low score, high NES)'                 AS section,
    p.subreddit,
    LEFT(p.title, 70)                                      AS title,
    p.score,
    p.num_comments,
    p.upvote_ratio,
    n.nes,
    n.engagement_tier
FROM nes_scores n
JOIN posts p USING (post_id)
WHERE n.engagement_tier = 'high'
  AND p.score < 100
ORDER BY n.nes DESC, p.num_comments DESC
LIMIT 10;


-- ── 5. Funnel conversion rate: what % of posts reach each tier? ───────────────

WITH tier_counts AS (
    SELECT
        engagement_tier,
        COUNT(*) AS n
    FROM nes_scores
    GROUP BY engagement_tier
),
total AS (
    SELECT SUM(n) AS total FROM tier_counts
)
SELECT
    '5. CONVERSION RATES'                                  AS section,
    t.engagement_tier,
    t.n                                                    AS posts_in_tier,
    ROUND(t.n * 100.0 / total.total, 2)                   AS conversion_rate_pct,
    -- Cumulative: what % reach this tier or higher?
    ROUND(SUM(t.n) OVER (
        ORDER BY
            CASE t.engagement_tier
                WHEN 'high' THEN 1
                WHEN 'mid'  THEN 2
                WHEN 'low'  THEN 3
            END
    ) * 100.0 / total.total, 2)                           AS cumulative_pct
FROM tier_counts t
CROSS JOIN total
ORDER BY
    CASE t.engagement_tier
        WHEN 'high' THEN 1
        WHEN 'mid'  THEN 2
        WHEN 'low'  THEN 3
    END;