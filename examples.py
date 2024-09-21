examples = [
    {
        "input": "Show me the daily performance trends for the last 30 days",
        "accountId": "act_624496083171435",
        "query": """
WITH daily_performance AS (
    SELECT 
        DATE(FROM_UNIXTIME(ci.date_start)) AS date,
        SUM(ci.impressions) AS daily_impressions,
        SUM(ci.unique_clicks) AS daily_clicks,
        SUM(ci.spend) AS daily_spend,
        AVG(ci.unique_ctr) AS daily_ctr,
        AVG(ci.cpc) AS daily_cpc
    FROM campaign_insights ci
    JOIN campaigns c ON ci.campaign_key = c.campaign_key
    WHERE c.account_id = 'act_624496083171435'
        AND ci.date_start >= UNIX_TIMESTAMP('2024-08-01 00:00:00') - (5.5 * 3600)
    GROUP BY DATE(FROM_UNIXTIME(ci.date_start))
)
SELECT 
    date,
    daily_impressions,
    daily_clicks,
    daily_spend,
    daily_ctr,
    daily_cpc,
    SUM(daily_spend) OVER (ORDER BY date) AS running_total_spend
FROM daily_performance
            ORDER BY date;
        """,
    },
    {
        "input": "Analyze the performance of different optimization goals across all ad sets in the past month",
        "accountId": "act_624496083171435",
        "query": """
SELECT 
    ads.optimization_goal,
    COUNT(DISTINCT ads.adset_key) AS adset_count,
    SUM(asi.impressions) AS total_impressions,
    SUM(asi.unique_clicks) AS total_clicks,
    SUM(asi.spend) AS total_spend,
    AVG(asi.unique_ctr) AS avg_ctr,
    AVG(asi.cpm) AS avg_cpm,
    AVG(asi.cpc) AS avg_cpc,
    SUM(asi.reach) AS total_reach,
    AVG(asi.frequency) AS avg_frequency
FROM ad_sets ads
JOIN ad_set_insights asi ON ads.adset_key = asi.adset_key
WHERE ads.campaign_key IN (
    SELECT campaign_key 
    FROM campaigns 
    WHERE account_id = 'act_624496083171435'
)
AND asi.date_start >= UNIX_TIMESTAMP('2024-08-01 00:00:00') - (5.5 * 3600)
GROUP BY ads.optimization_goal
ORDER BY avg_ctr DESC, avg_cpc ASC;
        """,
    },
    {
        "input": "What are the top 10 best performing campaign in past 30 days",
        "accountId": "act_624496083171435",
        "query": """
SELECT 
    c.name AS campaign_name,
    c.objective,
    SUM(ci.impressions) AS total_impressions,
    SUM(ci.unique_clicks) AS total_clicks,
    SUM(ci.spend) AS total_spend,
    AVG(ci.unique_ctr) AS avg_ctr,
    AVG(ci.cpm) AS avg_cpm,
    AVG(ci.cpc) AS avg_cpc,
    COUNT(DISTINCT DATE(FROM_UNIXTIME(ci.date_start))) AS days_run
FROM campaigns c
JOIN campaign_insights ci ON c.campaign_key = ci.campaign_key
WHERE c.account_id = 'act_624496083171435'
    AND ci.date_start >= UNIX_TIMESTAMP('2024-08-01 00:00:00') - (5.5 * 3600)
GROUP BY  c.name, c.objective
ORDER BY avg_ctr DESC , avg_cpm ASC 
LIMIT 10;
""",
    },
    {
        "input": "Provide a comparison of campaign performance grouped by campaign objectives for the past month.",
        "accountId": "act_624496083171435",
        "query": """
SELECT 
    c.objective,
    COUNT(DISTINCT c.name) AS campaign_count,
    SUM(ci.impressions) AS total_impressions,
    SUM(ci.clicks) AS total_clicks,
    SUM(ci.spend) AS total_spend,
    AVG(ci.ctr) AS average_ctr,
    AVG(ci.cpc) AS average_cpc,
    AVG(ci.cpm) AS average_cpm
FROM campaigns c
JOIN campaign_insights ci ON c.campaign_key = ci.campaign_key
WHERE c.account_id = 'act_624496083171435'
    AND ci.date_start >= UNIX_TIMESTAMP('2024-08-01 00:00:00') - (5.5 * 3600)
GROUP BY c.objective
            ORDER BY total_spend DESC;
        """,
    },
    {
        "input": "Whats the total ad spend by date in the past 7 days?",
        "accountId": "act_624496083171435",
        "query": """
SELECT
  DATE(FROM_UNIXTIME(date_start)) AS date,
  SUM(spend) AS total_spend
FROM ad_insights
WHERE
  date_start >= UNIX_TIMESTAMP('2024-08-01 00:00:00') - (5.5 * 3600)
GROUP BY DATE(FROM_UNIXTIME(date_start));
        """,
    },
    {
        "input": "Whats the total spend for campaigns in the past 7 days?",
        "accountId": "act_624496083171435",
        "query": """
SELECT
  c.name AS campaign_name,
  SUM(ci.spend) AS total_spend
FROM campaigns c
JOIN campaign_insights ci ON c.campaign_key = ci.campaign_key
WHERE
  c.account_id = 'act_624496083171435'
  AND ci.date_start >= UNIX_TIMESTAMP('2024-08-01 00:00:00') - (5.5 * 3600)
GROUP BY c.name
ORDER BY total_spend DESC;
        """,
    },
    {
        "input": "Which are our top 10 performing ad sets in terms of conversions over the last 30 days, and what's their average cost per conversion?",
        "accountId": "act_624496083171435",
        "query": """
SELECT
  c.name AS campaign_name,
  SUM(ci.impressions) AS total_impressions,
  SUM(ci.unique_clicks) AS total_clicks,
  SUM(ci.spend) AS total_spend,
  AVG(ci.cost_per_inline_link_click)as cpc,
  SUM(
    CAST(
      JSON_EXTRACT(
        JSON_EXTRACT(ci.conversions, '$[0]'),
        '$.value'
      ) AS DECIMAL(10,2)
    )
  ) AS total_conversions,
  AVG(
    CAST(
      JSON_EXTRACT(
        JSON_EXTRACT(ci.cost_per_conversion, '$[0]'),
        '$.value'
      ) AS DECIMAL(10,2)
    )
  ) AS avg_cost_per_conversion
FROM campaigns c
JOIN campaign_insights ci ON c.campaign_key = ci.campaign_key
WHERE
  c.account_id = 'act_624496083171435'
  AND ci.date_start >= UNIX_TIMESTAMP('2024-08-01 00:00:00') - (5.5 * 3600)
GROUP BY c.name
ORDER BY total_conversions DESC
LIMIT 10;
        """,
    },
]
