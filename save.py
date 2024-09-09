import pymysql
import json
import os
from decimal import Decimal
from dotenv import load_dotenv
from datetime import datetime
from dateutil import parser
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.WARNING)

def create_connection():
    timeout = 10
    return pymysql.connect(
        charset="utf8mb4",
        connect_timeout=timeout,
        cursorclass=pymysql.cursors.DictCursor,
        db=os.getenv("AIVEN_DATABASE", "defaultdb"),
        host=os.getenv("AIVEN_HOST"),
        password=os.getenv("AIVEN_PASSWORD"),
        read_timeout=timeout,
        port=int(os.getenv("AIVEN_PORT")),
        user=os.getenv("AIVEN_USER"),
        write_timeout=timeout,
    )

def to_epoch(date_string):
    return int(parser.parse(date_string).timestamp())


def save_campaigns(cursor):
    with open("metrics/campaigns.json", "r") as file:
        campaigns = json.load(file)

    for campaign in campaigns:
        sql = """
        INSERT INTO campaigns (id, name, account_id, objective, status, start_time, daily_budget, bid_strategy, currency, platform)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            campaign["id"],
            campaign["name"],
            campaign["account_id"],
            campaign["objective"],
            campaign["status"],
            to_epoch(campaign["start_time"]),
            Decimal(campaign["daily_budget"]) / 100,
            campaign.get("bid_strategy"),
            campaign["account_currency"],
            "facebook",
        )
        cursor.execute(sql, values)


def save_campaign_insights(cursor):
    with open("metrics/campaigns.json", "r") as file:
        campaigns = json.load(file)

    for campaign in campaigns:
        # Check if campaign exists
        cursor.execute("SELECT id FROM campaigns WHERE id = %s", (campaign["id"],))
        if not cursor.fetchone():
            logging.warning(f"Skipping insights for campaign {campaign['id']}: Campaign not found")
            continue

        if "insights" in campaign:
            for insight in campaign["insights"]:
                sql = """
                INSERT INTO campaign_insights 
                (campaign_id, impressions, clicks, spend, reach, frequency, ctr, cpm, cpp, cpc, unique_clicks, unique_ctr, date_start, date_stop)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    campaign["id"],
                    insight.get("impressions"),
                    insight.get("clicks"),
                    insight.get("spend"),
                    insight.get("reach"),
                    insight.get("frequency"),
                    insight.get("ctr"),
                    insight.get("cpm"),
                    insight.get("cpp"),
                    insight.get("cpc"),
                    insight.get("unique_clicks"),
                    insight.get("unique_ctr"),
                    to_epoch(insight["date_start"]),
                    to_epoch(insight["date_stop"])
                )
                cursor.execute(sql, values)
                insight_id = cursor.lastrowid

                if "actions" in insight:
                    for action in insight["actions"]:
                        cursor.execute(
                            """
                            INSERT INTO campaign_actions (campaign_insight_id, action_type, value)
                            VALUES (%s, %s, %s)
                            """,
                            (insight_id, action["action_type"], action["value"])
                        )


def update_daily_budgets(cursor):
    try:
        # Read the campaigns from the JSON file
        with open("data/campaigns2.json", "r") as file:
            campaigns = json.load(file)

        # Update each campaign's daily_budget
        for campaign in campaigns:
            if "daily_budget" in campaign:
                new_budget = Decimal(campaign["daily_budget"]) / 100
                update_sql = """
                UPDATE campaigns 
                SET daily_budget = %s 
                WHERE id = %s
                """
                cursor.execute(update_sql, (new_budget, campaign["id"]))

        print("Daily budgets updated successfully")
    except Exception as e:
        print(f"Error updating daily budgets: {e}")


def save_ads(cursor):
    with open("data/ads2.json", "r") as file:
        ads = json.load(file)

    for ad in ads:
        adset_id = ad.get("adset_id")
        
        # Check if adset_id exists in ad_sets table
        if adset_id:
            cursor.execute("SELECT id FROM ad_sets WHERE id = %s", (adset_id,))
            if not cursor.fetchone():
                logging.warning(f"Skipping ad {ad['id']}: AdSet {adset_id} not found")
                continue

        sql = """
        INSERT INTO ads (id, name, status, campaign_id, adset_id, currency, account_id, platform)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            ad["id"],
            ad["name"],
            ad["status"],
            ad["campaign"]["id"],
            adset_id,
            ad.get("account_currency"),
            ad.get("account_id"),
            "facebook",
        )
        cursor.execute(sql, values)

def save_ad_creatives(cursor):
    with open("data/ads2.json", "r") as file:
        ads = json.load(file)

    for ad in ads:
        # Check if ad exists
        cursor.execute("SELECT id FROM ads WHERE id = %s", (ad["id"],))
        if not cursor.fetchone():
            logging.warning(f"Skipping creative for ad {ad['id']}: Ad not found")
            continue

        if "creative" in ad and "_data" in ad["creative"]:
            creative = ad["creative"]["_data"]
            sql = """
            INSERT INTO ad_creatives (id, ad_id, name, title, call_to_action_type)
            VALUES (%s, %s, %s, %s, %s)
            """
            values = (
                creative["id"],
                ad["id"],
                creative.get("name"),
                creative.get("title"),
                creative.get("call_to_action_type")
            )
            cursor.execute(sql, values)

def save_ad_insights(cursor):
    with open("data/ads2.json", "r") as file:
        ads = json.load(file)

    for ad in ads:
        # Check if ad exists
        cursor.execute("SELECT id FROM ads WHERE id = %s", (ad["id"],))
        if not cursor.fetchone():
            logging.warning(f"Skipping insights for ad {ad['id']}: Ad not found")
            continue

        if "insights" in ad:
            for insight in ad["insights"]:
                sql = """
                INSERT INTO ad_insights 
                (ad_id, impressions, clicks, spend, reach, frequency, ctr, cpm, cpp, date_start, date_stop)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    ad["id"],
                    insight.get("impressions"),
                    insight.get("clicks"),
                    insight.get("spend"),
                    insight.get("reach"),
                    insight.get("frequency"),
                    insight.get("ctr"),
                    insight.get("cpm"),
                    insight.get("cpp"),
                    to_epoch(insight["date_start"]),
                    to_epoch(insight["date_stop"])
                )
                cursor.execute(sql, values)
                insight_id = cursor.lastrowid

                if "actions" in insight:
                    for action in insight["actions"]:
                        cursor.execute(
                            """
                            INSERT INTO ad_actions (ad_id, action_type, value)
                            VALUES (%s, %s, %s)
                            """,
                            (ad["id"], action["action_type"], action["value"])
                        )


def update_ad_creative_body(cursor):
    with open("data/ads2.json", "r") as file:
        ads = json.load(file)

    for ad in ads:
        # Check if ad exists
        cursor.execute("SELECT id FROM ads WHERE id = %s", (ad["id"],))
        if not cursor.fetchone():
            logging.warning(f"Skipping creative for ad {ad['id']}: Ad not found")
            continue

        if "creative" in ad and "_data" in ad["creative"]:
            creative = ad["creative"]["_data"]
            sql = """
            UPDATE ad_creatives
            SET body = %s
            WHERE id = %s
            """
            values = (creative.get("body"), creative["id"])
            cursor.execute(sql, values)


def save_ad_sets(cursor):
    with open("data/adsets3.json", "r") as file:
        ad_sets = json.load(file)

    for ad_set in ad_sets:
        sql = """
        INSERT INTO ad_sets (id, name, campaign_id, status, start_time, daily_budget, bid_strategy, billing_event, optimization_goal)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            ad_set["id"],
            ad_set["name"],
            ad_set["campaign_id"],
            ad_set["status"],
            to_epoch(ad_set["start_time"]),
            Decimal(ad_set["daily_budget"]) / 100 if "daily_budget" in ad_set else None,
            ad_set.get("bid_strategy"),
            ad_set["billing_event"],
            ad_set["optimization_goal"]
        )
        cursor.execute(sql, values)


def save_ad_set_insights(cursor):
    with open("metrics/adsets.json", "r") as file:
        ad_sets = json.load(file)

    for ad_set in ad_sets:
        # Check if ad set exists
        cursor.execute("SELECT id FROM ad_sets WHERE id = %s", (ad_set["id"],))
        if not cursor.fetchone():
            logging.warning(
                f"Skipping insights for ad set {ad_set['id']}: Ad set not found"
            )
            continue

        if "insights" in ad_set:
            for insight in ad_set["insights"]:
                sql = """
                INSERT INTO ad_set_insights 
                (ad_set_id, impressions, clicks, spend, reach, frequency, ctr, cpm, cpp, cpc, unique_clicks, unique_ctr, date_start, date_stop, age)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    ad_set["id"],
                    int(insight.get("impressions", 0)),
                    int(insight.get("clicks", 0)),
                    float(insight.get("spend", 0)),
                    int(insight.get("reach", 0)),
                    float(insight.get("frequency", 0)),
                    float(insight.get("ctr", 0)),
                    float(insight.get("cpm", 0)),
                    float(insight.get("cpp", 0)),
                    float(insight.get("cpc", 0)),
                    int(insight.get("unique_clicks", 0)),
                    float(insight.get("unique_ctr", 0)),
                    to_epoch(insight["date_start"]),
                    to_epoch(insight["date_stop"]),
                    insight.get("age")
                )
                cursor.execute(sql, values)


def update_ad_insight_new_fields(cursor):
    with open("metrics/ads.json", "r") as file:
        ads = json.load(file)

    for ad in ads:
        if "insights" in ad:
            for insight in ad["insights"]:
                sql = """
                UPDATE ad_insights
                SET hook_rate = %s, cost_per_action_type = %s, video_avg_time_watched_actions = %s, 
                    website_ctr = %s, video_completion_rate = %s, cost_per_unique_click = %s, outbound_clicks = %s
                WHERE ad_id = %s AND date_start = %s AND date_stop = %s
                """
                values = (
                    insight.get("hook_rate"),
                    json.dumps(insight.get("cost_per_action_type")),
                    json.dumps(insight.get("video_avg_time_watched_actions")),
                    json.dumps(insight.get("website_ctr")),
                    insight.get("video_completion_rate"),
                    insight.get("cost_per_unique_click"),
                    json.dumps(insight.get("outbound_clicks")),
                    ad["id"],
                    to_epoch(insight["date_start"]),
                    to_epoch(insight["date_stop"]),
                )
                cursor.execute(sql, values)


def update_ad_effective_status(cursor):
    with open("metrics/ads.json", "r") as file:
        ads = json.load(file)

    for ad in ads:
        sql = """
        UPDATE ads
        SET effective_status = %s
        WHERE id = %s
        """
        values = (ad.get("effective_status"), ad["id"])
        cursor.execute(sql, values)


def update_ad_set_insight_new_fields(cursor):
    with open("metrics/adsets.json", "r") as file:
        ad_sets = json.load(file)

    for ad_set in ad_sets:
        if "insights" in ad_set:
            for insight in ad_set["insights"]:
                sql = """
                UPDATE ad_set_insights
                SET conversions = %s, age = %s
                WHERE ad_set_id = %s AND date_start = %s AND date_stop = %s
                """
                values = (
                    json.dumps(insight.get("conversions")),
                    insight.get("age"),
                    ad_set["id"],
                    to_epoch(insight["date_start"]),
                    to_epoch(insight["date_stop"]),
                )
                cursor.execute(sql, values)


def update_ad_set_targeting(cursor):
    with open("data/adsets3.json", "r") as file:
        ad_sets = json.load(file)

    for ad_set in ad_sets:
        sql = """
        UPDATE ad_sets
        SET targeting = %s
        WHERE id = %s
        """
        values = (
            json.dumps(ad_set["targeting"]),
            ad_set["id"],
        )
        cursor.execute(sql, values)


def save_data():
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # save_campaigns(cursor)
        # save_campaign_insights(cursor)
        # save_ad_sets(cursor)
        save_ad_set_insights(cursor)
        # save_ads(cursor)
        save_ad_creatives(cursor)
        # save_ad_insights(cursor)
        # update_daily_budgets(cursor)
        # update_ad_effective_status(cursor)
        # update_ad_set_insight_new_fields(cursor)
        # update_ad_insight_new_fields(cursor)
        connection.commit()
        print("Data saved successfully")
    except pymysql.err.IntegrityError as e:
        connection.rollback()
        print(f"Integrity Error: {e}")
    except Exception as e:
        connection.rollback()
        print(f"Error saving data: {e}")
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    save_data()
