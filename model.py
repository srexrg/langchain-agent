import os
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from operator import itemgetter

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Database connection
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
    schema=db_name,
    view_support=True,
)
print(db.dialect)
print(db.get_usable_table_names())
print(db.table_info)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
generate_query_chain = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)


answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, provide a comprehensive answer following the guidelines below:

You are an expert Meta ads analyst and SQL query specialist. Your task is to interpret user questions about Meta ad performance, analyze SQL query results (including multi-row and multi-column data), and provide comprehensive, data-driven answers.

Follow these steps:
1. Analyze the user's question to understand the core topic and intent.
2. Carefully examine the SQL result, noting the number of rows and columns.
3. If the result is tabular (multiple rows/columns):
   a) Summarize the overall structure of the data (e.g., "The result shows data for 5 ads across 3 metrics").
   b) Identify and highlight key trends or patterns in the data.
   c) Mention the top 3-5 rows or most significant data points, providing context.
   d) Compare and contrast different rows or columns as relevant.
4. Interpret the query results, focusing on key Meta advertising metrics (e.g., CTR, CPC, ROAS, Frequency, Reach).
5. Provide a clear, actionable answer structured as follows:
   a) Summary of findings
   b) Detailed analysis with specific metrics and comparisons
   c) Performance insights and their implications
   d) At least two actionable recommendations based on the data
   e) Suggestions for follow-up analyses or questions
6. Relate your analysis to common Meta advertising objectives (e.g., awareness, consideration, conversion).
7. Consider the impact on different parts of the advertising funnel.

Remember to:
- Use ALL the data available in the query results, not just the top row.
- For tabular data, provide a holistic interpretation that covers the entire dataset.
- Clearly state any assumptions or limitations in your analysis.
- Maintain a professional yet conversational tone.
- Prioritize accuracy, relevance, and actionable insights.
- Use Meta-specific terminology where appropriate (e.g., ad sets, campaigns, placements).
- Consider the broader context of the Meta ads ecosystem (e.g., algorithm learning, audience saturation).
- If the data spans a time period, note any temporal trends or changes.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer: """
)

rephrase_answer = answer_prompt | llm | StrOutputParser()

chain = (
    RunnablePassthrough.assign(query=generate_query_chain).assign(
        result=itemgetter("query") | execute_query
     )
     | rephrase_answer
 )

chain.invoke({"question": "Ad with the most clicks?"})

# %%
examples = [
    {
        "input": "What are the top 5 campaigns by ROI in the last 30 days, considering all conversion types?",
        "accountId": "act_624496083171435",
        "query": """
            WITH campaign_performance AS (
                SELECT 
                    c.id,
                    c.name,
                    SUM(ci.spend) AS total_spend,
                    SUM(cc.value) AS total_conversions
                FROM campaigns c
                JOIN campaign_insights ci ON c.id = ci.campaign_id
                LEFT JOIN campaign_conversions cc ON ci.id = cc.campaign_insight_id
                WHERE c.account_id = 'act_624496083171435'
                    AND ci.date_start >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                GROUP BY c.id, c.name
            )
            SELECT 
                name AS campaign_name,
                total_spend,
                total_conversions,
                (total_conversions - total_spend) / total_spend * 100 AS roi_percentage
            FROM campaign_performance
            WHERE total_spend > 0
            ORDER BY roi_percentage DESC
            LIMIT 5;
        """,
    },
    {
        "input": "Compare the performance of different ad creative types across all campaigns in terms of CTR, CPC, and conversion rate in the last quarter",
        "accountId": "act_624496083171435",
        "query": """
            WITH ad_performance AS (
                SELECT 
                    ac.call_to_action_type,
                    SUM(ai.impressions) AS total_impressions,
                    SUM(ai.clicks) AS total_clicks,
                    SUM(ai.spend) AS total_spend,
                    SUM(cc.value) AS total_conversions
                FROM ad_creatives ac
                JOIN ads a ON ac.ad_id = a.id
                JOIN ad_insights ai ON a.id = ai.ad_id
                JOIN campaigns c ON a.campaign_id = c.id
                LEFT JOIN campaign_insights ci ON c.id = ci.campaign_id
                LEFT JOIN campaign_conversions cc ON ci.id = cc.campaign_insight_id
                WHERE c.account_id = 'act_624496083171435'
                    AND ai.date_start >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
                GROUP BY ac.call_to_action_type
            )
            SELECT 
                call_to_action_type,
                (total_clicks / NULLIF(total_impressions, 0)) * 100 AS ctr,
                total_spend / NULLIF(total_clicks, 0) AS cpc,
                (total_conversions / NULLIF(total_clicks, 0)) * 100 AS conversion_rate
            FROM ad_performance
            ORDER BY conversion_rate DESC;
        """,
    },
    {
        "input": "Analyze the daily budget utilization and performance trends for each campaign objective over the last 2 weeks",
        "accountId": "act_624496083171435",
        "query": """
            WITH daily_performance AS (
                SELECT 
                    c.id AS campaign_id,
                    c.name AS campaign_name,
                    c.objective,
                    c.daily_budget,
                    ci.date_start,
                    SUM(ci.spend) AS daily_spend,
                    SUM(ci.impressions) AS daily_impressions,
                    SUM(ci.clicks) AS daily_clicks,
                    SUM(cc.value) AS daily_conversions
                FROM campaigns c
                JOIN campaign_insights ci ON c.id = ci.campaign_id
                LEFT JOIN campaign_conversions cc ON ci.id = cc.campaign_insight_id
                WHERE c.account_id = 'act_624496083171435'
                    AND ci.date_start >= DATE_SUB(CURDATE(), INTERVAL 2 WEEK)
                GROUP BY c.id, c.name, c.objective, c.daily_budget, ci.date_start
            )
            SELECT 
                campaign_name,
                objective,
                date_start,
                daily_budget,
                daily_spend,
                (daily_spend / daily_budget) * 100 AS budget_utilization_percentage,
                daily_impressions,
                daily_clicks,
                daily_conversions,
                (daily_clicks / NULLIF(daily_impressions, 0)) * 100 AS daily_ctr,
                daily_spend / NULLIF(daily_clicks, 0) AS daily_cpc,
                (daily_conversions / NULLIF(daily_clicks, 0)) * 100 AS daily_conversion_rate
            FROM daily_performance
            ORDER BY campaign_name, date_start;
        """,
    },
    {
        "input": "Identify underperforming ads across all campaigns based on a composite score of CTR, CPC, and conversion rate, compared to campaign averages",
        "accountId": "act_624496083171435",
        "query": """
            WITH ad_metrics AS (
                SELECT 
                    a.id AS ad_id,
                    a.name AS ad_name,
                    a.campaign_id,
                    SUM(ai.impressions) AS impressions,
                    SUM(ai.clicks) AS clicks,
                    SUM(ai.spend) AS spend,
                    SUM(cc.value) AS conversions
                FROM ads a
                JOIN ad_insights ai ON a.id = ai.ad_id
                JOIN campaigns c ON a.campaign_id = c.id
                LEFT JOIN campaign_insights ci ON c.id = ci.campaign_id
                LEFT JOIN campaign_conversions cc ON ci.id = cc.campaign_insight_id
                WHERE c.account_id = 'act_624496083171435'
                GROUP BY a.id, a.name, a.campaign_id
            ),
            campaign_avg_metrics AS (
                SELECT 
                    campaign_id,
                    AVG(clicks / NULLIF(impressions, 0)) AS avg_ctr,
                    AVG(spend / NULLIF(clicks, 0)) AS avg_cpc,
                    AVG(conversions / NULLIF(clicks, 0)) AS avg_cvr
                FROM ad_metrics
                GROUP BY campaign_id
            )
            SELECT 
                am.ad_name,
                am.campaign_id,
                (am.clicks / NULLIF(am.impressions, 0)) AS ctr,
                (am.spend / NULLIF(am.clicks, 0)) AS cpc,
                (am.conversions / NULLIF(am.clicks, 0)) AS cvr,
                cam.avg_ctr,
                cam.avg_cpc,
                cam.avg_cvr,
                (
                    ((am.clicks / NULLIF(am.impressions, 0)) / NULLIF(cam.avg_ctr, 0)) +
                    (cam.avg_cpc / NULLIF((am.spend / NULLIF(am.clicks, 0)), 0)) +
                    ((am.conversions / NULLIF(am.clicks, 0)) / NULLIF(cam.avg_cvr, 0))
                ) / 3 AS performance_score
            FROM ad_metrics am
            JOIN campaign_avg_metrics cam ON am.campaign_id = cam.campaign_id
            WHERE am.impressions > 0
            ORDER BY performance_score ASC
            LIMIT 10;
        """,
    },
]

# %%
example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}\nAccount ID: {accountId}\nSQL Query:"), ("ai", "{query}")]
)

vectorstore = Chroma()
vectorstore.delete_collection()
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    vectorstore,
    k=2,
    input_keys=["input"],
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input", "top_k"],
)

# %%
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a MySQL expert. Given an input question and an account ID, create a syntactically correct MySQL query to run. Always include the account_id in your WHERE clause for filtering.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions, account IDs, and their corresponding SQL queries."),
        few_shot_prompt,
        ("human", "{input}\nAccount ID: {accountId}"),
    ]
)
generate_query = create_sql_query_chain(llm, db, final_prompt)
chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | execute_query
    )
    | rephrase_answer
)


def process_question(question, account_id):
    generated_query = generate_query.invoke(
        {
            "question": question,
            "accountId": account_id,
        }
    )
    clean_query = (
        generated_query.strip().replace("```sql", "").replace("```", "").strip()
    )

    try:
        result = execute_query.invoke(clean_query)
        answer = rephrase_answer.invoke(
            {"question": question, "query": clean_query, "result": result}
        )
        return clean_query, answer
    except Exception as e:
        return clean_query, f"An error occurred: {str(e)}"


# Example usage
if __name__ == "__main__":
    question = "What are the campaigns that we should increase the budget of in last 5 months? by how much? and why do you think we should increase the budget of that particular campaign?"
    account_id = "act_624496083171435"

    query, answer = process_question(question, account_id)
    print("SQL Query:")
    print(query)
    print("\nFinal Answer:")
    print(answer)