CREATE TABLE campaigns (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    account_id VARCHAR(50),
    objective VARCHAR(50),
    status VARCHAR(20),
    start_time BIGINT,
    daily_budget DECIMAL(10, 2),
    bid_strategy VARCHAR(50),
    currency VARCHAR(3);
    platform VARCHAR(50)
);

-- Campaign Insights table
CREATE TABLE campaign_insights (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    campaign_id BIGINT, //index
    impressions INT,
    clicks INT,
    spend DECIMAL(10, 2),
    reach INT,
    frequency DECIMAL(10, 6),
    ctr DECIMAL(10, 6),
    cpm DECIMAL(10, 6),
    cpp DECIMAL(10, 6),
    cpc DECIMAL(10, 6),
    unique_clicks INT,
    unique_ctr DECIMAL(10, 6),
    date_start BIGINT,
    date_stop BIGINT,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id),
    INDEX (campaign_id)
);


CREATE TABLE campaign_actions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    campaign_insight_id BIGINT,
    action_type VARCHAR(100),
    value INT,
    FOREIGN KEY (campaign_insight_id) REFERENCES campaign_insights(id)
);

-- Campaign Conversions table
-- CREATE TABLE campaign_conversions (
--     id BIGINT AUTO_INCREMENT PRIMARY KEY,
--     campaign_insight_id BIGINT,
--     action_type VARCHAR(100),
--     value INT,
--     FOREIGN KEY (campaign_insight_id) REFERENCES campaign_insights(id)
-- );
CREATE TABLE ads (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    status VARCHAR(20),
    campaign_id BIGINT,
    adset_id BIGINT,
    currency VARCHAR(3);
    account_id VARCHAR(50),
    effective_status VARCHAR(20),
    platform VARCHAR(50)
    FOREIGN KEY (adset_id) REFERENCES ad_sets(id)  
);

-- Ad Creatives table
CREATE TABLE ad_creatives (
    id BIGINT,
    ad_id BIGINT,
    name VARCHAR(255),
    title VARCHAR(255),
    body TEXT,
    call_to_action_type VARCHAR(50),
    PRIMARY KEY (id, ad_id),
    FOREIGN KEY (ad_id) REFERENCES ads(id)
);

-- Ad Insights table
CREATE TABLE ad_insights (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ad_id BIGINT, //index
    impressions INT,
    clicks INT,
    spend DECIMAL(10, 2),
    reach INT,
    frequency DECIMAL(10, 6),
    ctr DECIMAL(10, 6),
    cpm DECIMAL(10, 6),
    cpp DECIMAL(10, 6),
    date_start BIGINT,
    hook_rate DECIMAL(10, 6),
    cost_per_action_type JSON,   
    video_avg_time_watched_actions JSON,
    website_ctr JSON,
    video_completion_rate DECIMAL(10, 6),
    cost_per_unique_click DECIMAL(10, 6), 
    outbound_clicks JSON,
    date_stop BIGINT,
    FOREIGN KEY (ad_id) REFERENCES ads(id),
    INDEX (ad_id)
);

CREATE TABLE ad_sets (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    campaign_id BIGINT,
    status VARCHAR(20),
    start_time BIGINT,
    daily_budget DECIMAL(10, 2),
    bid_strategy VARCHAR(50),
    billing_event VARCHAR(50),
    optimization_goal VARCHAR(50),
    targeting JSON,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
);

CREATE TABLE ad_set_insights (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ad_set_id BIGINT,
    impressions INT,
    clicks INT,
    spend DECIMAL(10, 2),
    reach INT,
    frequency DECIMAL(10, 6),
    ctr DECIMAL(10, 6),
    cpm DECIMAL(10, 6),
    cpp DECIMAL(10, 6),
    cpc DECIMAL(10, 6),
    unique_clicks INT,
    unique_ctr DECIMAL(10, 6),
    date_start BIGINT,
    date_stop BIGINT,
    age VARCHAR(10), 
    FOREIGN KEY (ad_set_id) REFERENCES ad_sets(id),
    INDEX (ad_set_id)  -- Added this line
);




//adsets

//adset_id->ad table 

//adset insights table

    //adset_id//index

