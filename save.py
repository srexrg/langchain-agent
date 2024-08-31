import pymysql
import json
import os
from decimal import Decimal
from dotenv import load_dotenv
from datetime import datetime
from dateutil import parser

load_dotenv()


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


def save_campaign_data(cursor):
    with open("data/campaigns.json", "r") as file:
        campaign_data = json.load(file)

    for campaign in campaign_data:
        # Convert daily_budget from string to Decimal (assuming it's in INR)
        daily_budget = (
            Decimal(campaign["daily_budget"]) / 100
            if "daily_budget" in campaign
            else None
        )

        # Convert start_time to MySQL-compatible format
        start_time = parser.isoparse(campaign['start_time']).strftime('%Y-%m-%d %H:%M:%S')

        # Update the SQL query
        sql = """
        INSERT INTO campaigns (id, name, account_id, objective, status, start_time, daily_budget, bid_strategy, platform)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            campaign["id"],
            campaign["name"],
            campaign.get("account_id", None),
            campaign["objective"],
            campaign["status"],
            start_time,
            daily_budget,
            campaign.get("bid_strategy", None),
            "facebook",
        )

        cursor.execute(sql, values)

        if "insights" in campaign:
            for insight in campaign["insights"]:
                cursor.execute(
                    """
                INSERT INTO campaign_insights 
                (campaign_id, impressions, clicks, spend, reach, frequency, ctr, cpm, cpp, cpc, 
                unique_clicks, unique_ctr, date_start, date_stop)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        campaign["id"],
                        insight["impressions"],
                        insight["clicks"],
                        insight["spend"],
                        insight["reach"],
                        insight["frequency"],
                        insight["ctr"],
                        insight["cpm"],
                        insight["cpp"],
                        insight["cpc"],
                        insight["unique_clicks"],
                        insight["unique_ctr"],
                        insight["date_start"],
                        insight["date_stop"],
                    ),
                )
                insight_id = cursor.lastrowid

                if "actions" in insight:
                    for action in insight["actions"]:
                        cursor.execute(
                            """
                        INSERT INTO campaign_actions (campaign_insight_id, action_type, value)
                        VALUES (%s, %s, %s)
                        """,
                            (insight_id, action["action_type"], action["value"]),
                        )

                if "conversions" in insight:
                    for conversion in insight["conversions"]:
                        cursor.execute(
                            """
                        INSERT INTO campaign_conversions (campaign_insight_id, action_type, value)
                        VALUES (%s, %s, %s)
                        """,
                            (
                                insight_id,
                                conversion["action_type"],
                                conversion["value"],
                            ),
                        )


def save_ad_data(cursor):
    with open("data/ads.json", "r") as file:
        ad_data = json.load(file)

    for ad in ad_data:
        cursor.execute(
            """
        INSERT INTO ads (id, name, status, campaign_id, account_id, platform)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
            (
                ad["id"],
                ad["name"],
                ad["status"],
                ad["campaign"]["id"],
                ad["account_id"],
                "Facebook",
            ),
        )

        if "creative" in ad and "_data" in ad["creative"]:
            creative = ad["creative"]["_data"]
            cursor.execute(
                """
            INSERT INTO ad_creatives (id, ad_id, name, title, call_to_action_type)
            VALUES (%s, %s, %s, %s, %s)
            """,
                (
                    creative["id"],
                    ad["id"],
                    creative.get("name", None),
                    creative.get("title", None),
                    creative.get("call_to_action_type", None),
                ),
            )

        if "insights" in ad:
            for insight in ad["insights"]:
                cursor.execute(
                    """
                INSERT INTO ad_insights 
                (ad_id, impressions, clicks, spend, reach, frequency, ctr, cpm, cpp, date_start, date_stop)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        ad["id"],
                        insight["impressions"],
                        insight["clicks"],
                        insight["spend"],
                        insight["reach"],
                        insight["frequency"],
                        insight["ctr"],
                        insight["cpm"],
                        insight["cpp"],
                        insight["date_start"],
                        insight["date_stop"],
                    ),
                )
                insight_id = cursor.lastrowid


def save_data():
    connection = create_connection()
    cursor = connection.cursor()

    try:
        save_campaign_data(cursor)
        save_ad_data(cursor)
        connection.commit()
        print("Data saved successfully")
    except Exception as e:
        connection.rollback()
        print(f"Error saving data: {e}")
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    save_data()
