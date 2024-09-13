import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from operator import itemgetter
from langchain.tools.base import BaseTool

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Get database credentials from environment variables
db_user = os.getenv("AIVEN_USER")
db_password = os.getenv("AIVEN_PASSWORD")
db_host = os.getenv("AIVEN_HOST")
db_port = os.getenv("AIVEN_PORT")
db_name = os.getenv("AIVEN_DATABASE")

# Create database connection
db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
    schema=db_name,
    view_support=True,
)
print(db.dialect)
print(db.get_usable_table_names())
print(db.table_info)

# Initialize vector store for knowledge base
embedding_model = OpenAIEmbeddings()
vector_store = Chroma(persist_directory="./chunk", embedding_function=embedding_model)


# Define custom tool for knowledge base
class KnowledgeBaseTool(BaseTool):
    name = "Knowledge Base"
    description = "Use this tool to query the knowledge base for relevant context"

    def _run(self, query: str) -> str:
        results = vector_store.similarity_search(query, k=2)
        return "\n".join([doc.page_content for doc in results])

    async def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# New prompt for determining whether to query DB or use LLM knowledge
query_or_knowledge_prompt = PromptTemplate.from_template(
    """Given the following user question, determine if it requires querying the database or if it can be answered using general knowledge about Facebook advertising.

User question: {question}

If the question requires specific data from the user's ad account, respond with "QUERY_DB". If it can be answered with general knowledge about Facebook advertising, respond with "USE_LLM".

Response:"""
)

query_or_knowledge_chain = query_or_knowledge_prompt | llm | StrOutputParser()

# Function to answer general knowledge questions
def answer_general_question(question: str) -> str:
    general_knowledge_prompt = PromptTemplate.from_template(
        """You are an expert in Facebook advertising. Please answer the following question about Facebook advertising best practices, strategies, or general concepts:

Question: {question}

Answer:"""
    )
    general_knowledge_chain = general_knowledge_prompt | llm | StrOutputParser()
    return general_knowledge_chain.invoke({"question": question})

generate_query_chain = create_sql_query_chain(llm, db)

execute_query = QuerySQLDataBaseTool(db=db)
chain = generate_query_chain | execute_query

answer_prompt = PromptTemplate.from_template(
    """Conduct a thorough analysis of the user's Meta ad account performance, leveraging all available data sources. When information is limited or unclear, engage the user to gather more context. Follow these steps:

Data Collection and Integration:
a) Access and retrieve data from the user's connected Meta ad account(s)
b) If access is limited or data seems incomplete, ask the user:
"I'm having trouble accessing all your ad account data. Can you confirm you've granted full permissions? Are there any specific accounts or campaigns you want me to focus on?"
c) Collect data for all active and recently concluded campaigns (past 90 days)
d) If historical data is limited, ask:
"I can only see data for [X] days. Would you like me to analyze this period, or do you have a specific timeframe in mind?"
e) Gather relevant industry benchmarks from AdsNerd's curated knowledge base
f) If industry classification is unclear, ask:
"To provide relevant benchmarks, could you tell me which industry your business falls under?"
Key Metrics Analysis:
a) Calculate and analyze the following metrics:

Return on Ad Spend (ROAS)
Cost Per Acquisition (CPA)
Click-Through Rate (CTR)
Conversion Rate
Total Ad Spend
Impressions
Reach
Frequency
Relevance Score or Quality Ranking
b) Compare current period performance (last 30 days) with:
Previous period (30 days before last)
Same period last year (if data available)
Industry benchmarks
c) If any key metrics are missing or seem unusual, ask:
"I noticed [Metric X] is [missing/unusually high/low]. Is this expected, or has there been a recent change in your campaign strategy?"


Campaign Performance Breakdown:
a) Identify top 3 and bottom 3 performing campaigns based on primary objective (e.g., ROAS, Conversions)
b) Analyze performance trends for each campaign over time
c) Compare campaign performance to account averages and industry standards
d) Highlight any significant changes or anomalies in campaign performance
e) If campaign objectives are unclear, ask:
"What are the primary objectives for your top campaigns? This will help me provide more relevant insights."
Audience Analysis:
a) Evaluate performance across different audience segments
b) Identify best and worst-performing audiences
c) Analyze audience overlap and potential audience fatigue
d) Compare audience performance to industry trends and benchmarks
e) If audience data is limited, ask:
"I have limited information about your audience segments. Can you tell me about your target audience or any specific segments you're focusing on?"
Creative Performance Assessment:
a) Analyze performance of different ad formats (image, video, carousel, etc.)
b) Identify top-performing ad creatives across campaigns
c) Evaluate ad copy effectiveness and messaging themes
d) If available, compare creative approaches to visible competitor strategies from Ad Library
e) If creative data is limited, ask:
"I don't have much information about your ad creatives. Could you describe your main ad types and any recent changes in creative strategy?"
Budget and Bidding Evaluation:
a) Assess overall budget utilization and pacing
b) Identify campaigns that are overspending or underspending
c) Analyze the effectiveness of current bidding strategies
d) Compare budget efficiency to industry averages
e) If budget information is unclear, ask:
"Can you provide more details about your budget allocation strategy or any recent changes in spending?"
Placement Performance:
a) Evaluate ad performance across different placements (Feed, Stories, Audience Network, etc.)
b) Identify most and least effective placements
c) Compare placement performance to industry benchmarks
d) If placement data is limited, ask:
"I'm seeing limited data on ad placements. Are you targeting specific placements, or using automatic placements?"
Conversion Funnel Analysis:
a) Analyze the conversion path from impression to purchase
b) Identify drop-off points in the funnel
c) Compare funnel performance to industry standards
d) If conversion data is incomplete, ask:
"I'm having trouble seeing your full conversion funnel. Can you describe your typical customer journey from ad view to purchase?"
Key Insights and Recommendations:
a) Synthesize 5-7 key insights from the analysis
b) Prioritize insights based on potential impact on overall performance
c) Provide initial recommendations for improvement, referencing AdsNerd's knowledge base of best practices
d) Suggest areas that require deeper analysis, referencing other relevant AdsNerd modules
e) After presenting insights, ask:
"Do these insights align with your observations? Are there any areas you'd like me to explore further?"
Competitive Landscape (if Ad Library data is available):
a) Provide an overview of visible competitor ad strategies
b) Highlight any significant differences in approach or performance
c) If competitor data is unavailable, ask:
"I don't have access to competitor data. Could you share any insights about your main competitors' advertising strategies?"
Data Visualization:
a) Prepare clear, easy-to-understand charts or graphs for key metrics and trends
b) Ensure visualizations are descriptive and can be easily explained in text format
Summary and Next Steps:
a) Provide a concise summary of overall account health and performance
b) Outline recommended next steps for optimization
c) Suggest specific AdsNerd modules for deeper dives into areas of concern or opportunity
d) After presenting the summary, ask:
"Based on this overview, what aspects of your ad performance are you most interested in improving?"

Throughout the analysis, maintain a conversational tone and avoid jargon. Be prepared to explain any technical terms or concepts if the user requests clarification. Always specify the source of data, especially when referencing industry benchmarks or competitor insights.
If any critical data is missing or there are limitations in the analysis, clearly communicate this to the user and explain how it might impact the insights provided. Use the questions provided to gather more context and provide the most accurate and helpful analysis possible.
Remember, the goal is to provide the user with a clear, actionable understanding of their Meta ad performance and set the stage for more detailed optimization efforts using other AdsNerd modules.
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
examples = [
    {
        "input": "What are the top 5 ads by CTR and their spend in the last 30 days?",
        "accountId": "act_624496083171435",
        "query": """
            SELECT 
                a.name AS ad_name,
                ai.ctr,
                ai.impressions,
                ai.clicks,
                ai.spend,
                a.currency,
                FROM_UNIXTIME(ai.date_start) AS date_start,
                FROM_UNIXTIME(ai.date_stop) AS date_stop
            FROM ads a
            JOIN ad_insights ai ON a.id = ai.ad_id
            WHERE a.account_id = 'act_624496083171435'
                AND ai.date_start >= UNIX_TIMESTAMP(DATE_SUB(CURDATE(), INTERVAL 30 DAY))
            ORDER BY ctr DESC
            LIMIT 5;
        """,
    },
    # {
    #     "input": "Compare the performance of different campaign objectives in terms of spend and conversions",
    #     "accountId": "act_624496083171435",
    #     "query": """
    #         SELECT
    #             c.objective,
    #             SUM(ci.spend) AS total_spend,
    #             c.currency,
    #             SUM(ca.value) AS total_conversions,
    #             AVG(ci.ctr) AS avg_ctr,
    #             AVG(ci.cpm) AS avg_cpm
    #         FROM campaigns c
    #         JOIN campaign_insights ci ON c.id = ci.campaign_id
    #         LEFT JOIN campaign_actions ca ON ci.id = ca.campaign_insight_id
    #         WHERE c.account_id = 'act_624496083171435' AND ca.action_type='offsite_conversion.fb_pixel_custom'
    #         GROUP BY c.objective, c.currency
    #         ORDER BY total_spend DESC;
    #     """,
    # },
    {
        "input": "What is the daily performance trend of our top-spending campaign in the last week?",
        "accountId": "act_624496083171435",
        "query": """
            WITH top_campaign AS (
                SELECT campaign_id
                FROM campaign_insights
                WHERE date_start >= UNIX_TIMESTAMP(DATE_SUB(CURDATE(), INTERVAL 7 DAY))
                GROUP BY campaign_id
                ORDER BY SUM(spend) DESC
                LIMIT 1
            )
            SELECT 
                c.name AS campaign_name,
                FROM_UNIXTIME(ci.date_start) AS date_start,
                ci.impressions,
                ci.clicks,
                ci.spend,
                c.currency,
                ci.ctr,
                ci.cpm,
                ca.value AS conversions
            FROM campaigns c
            JOIN campaign_insights ci ON c.id = ci.campaign_id
            LEFT JOIN campaign_actions ca ON ci.id = ca.campaign_insight_id
            WHERE c.id = (SELECT campaign_id FROM top_campaign)
                AND c.account_id = 'act_624496083171435'
                AND ci.date_start >= UNIX_TIMESTAMP(DATE_SUB(CURDATE(), INTERVAL 7 DAY))
            ORDER BY ci.date_start;
        """,
    },
    # {
    #     "input": "Which ad creative type has the highest conversion rate and what's the total spend for each?",
    #     "accountId": "act_624496083171435",
    #     "query": """
    #         SELECT
    #             ac.call_to_action_type,
    #             SUM(ai.clicks) AS total_clicks,
    #             SUM(ca.value) AS total_conversions,
    #             SUM(ca.value) / NULLIF(SUM(ai.clicks), 0) * 100 AS conversion_rate,  // Avoid division by zero
    #             SUM(ai.spend) AS total_spend,
    #             a.currency
    #         FROM ad_creatives ac
    #         JOIN ads a ON ac.ad_id = a.id
    #         JOIN ad_insights ai ON a.id = ai.ad_id
    #         JOIN campaigns c ON a.campaign_id = c.id
    #         LEFT JOIN campaign_insights ci ON c.id = ci.campaign_id
    #         LEFT JOIN campaign_actions ca ON ci.id = ca.campaign_insight_id
    #         WHERE c.account_id = 'act_624496083171435'
    #         GROUP BY ac.call_to_action_type, a.currency
    #         ORDER BY conversion_rate DESC;
    #     """,
    # },
]

example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}\nAccount ID: {accountId}\nSQL Query:"), ("ai", "{query}")]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt, examples=examples, input_variables=["input"]
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
table_info = """
[
  {
    “id”: “entity_campaign”,
    “text”: “Campaign is a core entity in ad performance analysis. It represents the highest level of organization for ads.“,
    “category”: “Core Entities”,
    “related_entities”: [“Ad Set”, “Ad”],
    “table_dependencies”: [“campaigns”, “campaign_insights”]
  },
  {
    “id”: “entity_ad_set”,
    “text”: “Ad Set is a core entity that sits between Campaign and Ad. It often defines targeting and bidding strategies.“,
    “category”: “Core Entities”,
    “related_entities”: [“Campaign”, “Ad”],
    “table_dependencies”: [“ad_sets”, “ad_set_insights”]
  },
  {
    “id”: “entity_ad”,
    “text”: “Ad is the core entity representing individual advertisements shown to users.“,
    “category”: “Core Entities”,
    “related_entities”: [“Campaign”, “Ad Set”, “Ad Creative”],
    “table_dependencies”: [“ads”, “ad_insights”]
  },
  {
    “id”: “metric_impressions”,
    “text”: “Impressions: Number of times ads were shown to users.“,
    “category”: “Impression Metrics”,
    “sql_relevance”: “SUM aggregate function”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “metric_reach”,
    “text”: “Reach: Number of unique users who saw ads.“,
    “category”: “Impression Metrics”,
    “sql_relevance”: “COUNT DISTINCT or similar function”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “metric_frequency”,
    “text”: “Frequency: Average number of times each user saw ads. Calculated as Impressions / Reach.“,
    “category”: “Impression Metrics”,
    “sql_relevance”: “Division of aggregates”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “metric_clicks”,
    “text”: “Clicks: Total number of clicks on ads.“,
    “category”: “Click Metrics”,
    “sql_relevance”: “SUM aggregate function”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “metric_ctr”,
    “text”: “CTR (Click-Through Rate): Percentage of impressions that resulted in clicks. Calculated as (Clicks / Impressions) * 100.“,
    “category”: “Click Metrics”,
    “sql_relevance”: “Division of aggregates, percentage calculation”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “metric_spend”,
    “text”: “Spend: Total money spent on ads.“,
    “category”: “Cost Metrics”,
    “sql_relevance”: “SUM aggregate function”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “metric_cpm”,
    “text”: “CPM (Cost Per Mille): Cost per 1000 impressions. Calculated as (Spend / Impressions) * 1000.“,
    “category”: “Cost Metrics”,
    “sql_relevance”: “Division of aggregates, multiplication”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “metric_cpc”,
    “text”: “CPC (Cost Per Click): Cost per click. Calculated as Spend / Clicks.“,
    “category”: “Cost Metrics”,
    “sql_relevance”: “Division of aggregates”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “metric_conversions”,
    “text”: “Conversions: Desired actions taken (e.g., purchases, leads).“,
    “category”: “Action Metrics”,
    “sql_relevance”: “SUM aggregate function”,
    “table_dependencies”: [“ad_actions”, “campaign_actions”]
  },
  {
    “id”: “metric_video_completion_rate”,
    “text”: “Video Completion Rate: Percentage of video ad views that were completed.“,
    “category”: “Video Metrics”,
    “sql_relevance”: “Percentage calculation”,
    “table_dependencies”: [“ad_insights”]
  },
  {
    “id”: “metric_roi”,
    “text”: “ROI (Return on Investment): (Value of Conversions - Spend) / Spend”,
    “category”: “Calculated Metrics”,
    “sql_relevance”: “Complex calculation involving multiple metrics”,
    “table_dependencies”: [“ad_insights”, “ad_actions”, “campaign_insights”, “campaign_actions”]
  },
  {
    “id”: “dimension_time”,
    “text”: “Time dimension for analysis using date_start and date_stop fields.“,
    “category”: “Dimensions for Analysis”,
    “sql_relevance”: “Date filtering, GROUP BY clause”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “dimension_campaign”,
    “text”: “Campaign dimension for analysis using campaign_id and campaign_name.“,
    “category”: “Dimensions for Analysis”,
    “sql_relevance”: “JOIN operations, GROUP BY clause”,
    “table_dependencies”: [“campaigns”, “campaign_insights”]
  },
  {
    “id”: “analysis_pattern_time_series”,
    “text”: “Analyze performance over time using date_start and date_stop fields.“,
    “category”: “Common Analysis Patterns”,
    “sql_relevance”: “Time-based GROUP BY, window functions”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “analysis_pattern_entity_comparison”,
    “text”: “Compare performance between entities (campaigns, ad sets, ads).“,
    “category”: “Common Analysis Patterns”,
    “sql_relevance”: “JOINs, subqueries, window functions for ranking”,
    “table_dependencies”: [“campaigns”, “ad_sets”, “ads”, “ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “relationship_campaign_adset”,
    “text”: “Campaigns contain multiple Ad Sets. Use campaign_id in ad_sets table to establish relationship.“,
    “category”: “Key Relationships”,
    “sql_relevance”: “JOIN operations”,
    “table_dependencies”: [“campaigns”, “ad_sets”]
  },
  {
    “id”: “relationship_adset_ad”,
    “text”: “Ad Sets contain multiple Ads. Use ad_set_id in ads table to establish relationship.“,
    “category”: “Key Relationships”,
    “sql_relevance”: “JOIN operations”,
    “table_dependencies”: [“ad_sets”, “ads”]
  },
  {
    “id”: “optimization_cost_per_unique_click”,
    “text”: “Cost per unique click is an optimization metric to measure efficiency of ad spend.“,
    “category”: “Optimization Metrics”,
    “sql_relevance”: “Division of aggregates”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “advanced_concept_attribution”,
    “text”: “Attribution is implied by action_type in ad_actions and campaign_actions tables.“,
    “category”: “Advanced Concepts”,
    “sql_relevance”: “Filtering, JOIN operations”,
    “table_dependencies”: [“ad_actions”, “campaign_actions”]
  },
  {
    “id”: “aggregation_campaign_level”,
    “text”: “Aggregate data at the campaign level using campaigns and campaign_insights tables.“,
    “category”: “Data Aggregation Levels”,
    “sql_relevance”: “GROUP BY campaign_id, JOIN operations”,
    “table_dependencies”: [“campaigns”, “campaign_insights”]
  },
  {
    “id”: “constraint_date_range”,
    “text”: “Apply date range constraints using date_start and date_stop fields.“,
    “category”: “Common Constraints”,
    “sql_relevance”: “WHERE clause with date comparisons”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “visualization_time_series”,
    “text”: “Use time series visualization for trend analysis of metrics over time.“,
    “category”: “Visualization Considerations”,
    “sql_relevance”: “Time-based GROUP BY, ordering”,
    “table_dependencies”: [“ad_insights”, “ad_set_insights”, “campaign_insights”]
  },
  {
    “id”: “best_practice_cte”,
    “text”: “Use CTEs (Common Table Expressions) for complex queries to improve readability and maintainability.“,
    “category”: “Best Practices”,
    “subcategory”: “Query Construction and Optimization”,
    “sql_relevance”: “WITH clause in SQL”
  },
  {
    “id”: “best_practice_nullif”,
    “text”: “Always use NULLIF when dividing to avoid division by zero errors (e.g., NULLIF(clicks, 0)).“,
    “category”: “Best Practices”,
    “subcategory”: “Metric Calculation and Interpretation”,
    “sql_relevance”: “NULLIF function in divisions”
  },
  {
    “id”: “best_practice_date_filters”,
    “text”: “Always include date filters to ensure consistent time ranges across different metrics.“,
    “category”: “Best Practices”,
    “subcategory”: “Time-based Analysis”,
    “sql_relevance”: “WHERE clause with date comparisons”
  },
  {
    “id”: “best_practice_objective_alignment”,
    “text”: “Focus on metrics that align with the campaign objective (e.g., conversions for conversion campaigns, reach for awareness campaigns).“,
    “category”: “Best Practices”,
    “subcategory”: “Actionable Insights”,
    “sql_relevance”: “Conditional metric selection based on campaign.objective”
  }
]
"""

# user query to embedding -> search for similar examples -> use the context to generate query

# similarity search for docs 

# csv files for  storing the db schema





final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a MySQL expert. Given an input question, account ID, and additional context, create a syntactically correct MySQL query to run. Always include the account_id in your WHERE clause for filtering.

Here is the relevant table info: {table_info}

Important notes:
1. Always consider the currency when dealing with monetary values. The currency is stored in the 'currency' column of the 'ads' and 'campaigns' tables.
2. When comparing or aggregating monetary values, ensure they are in the same currency or use appropriate conversion rates.
3. Include the currency in your SELECT statement when querying monetary values.
Below are a number of examples of questions, account IDs, and their corresponding SQL queries.""",
        ),
        few_shot_prompt,
        ("human", "{input}\nAccount ID: {accountId}"),
    ]
)
generate_query = create_sql_query_chain(llm, db, final_prompt)


# Modify the process_question function
def process_question(question: str, account_id: str):
    # Determine whether to query DB or use LLM knowledge
    decision = query_or_knowledge_chain.invoke({"question": question})
    
    if decision == "QUERY_DB":
        generated_query = generate_query.invoke({
            "question": question,
            "accountId": account_id,
            "table_info": table_info,
        })
        clean_query = generated_query.strip().replace("```sql", "").replace("```", "").strip()

        try:
            result = execute_query.invoke(clean_query)
            answer = rephrase_answer.invoke({
                "question": question,
                "query": clean_query,
                "result": result,
            })
            return clean_query, answer
        except Exception as e:
            return clean_query, f"An error occurred: {str(e)}"
    elif decision == "USE_LLM":
        answer = answer_general_question(question)
        return None, answer
    else:
        return None, "Unable to determine how to process the question. Please try rephrasing your query."


# Example usage
if __name__ == "__main__":
    questions = [
        "What was the best ctr"
    ]
    account_id = "act_624496083171435"

    for question in questions:
        print(f"\nQuestion: {question}")
        query, answer = process_question(question, account_id)
        if query:
            print("Generated SQL query:")
            print(query)
        print("\nAnswer:")
        print(answer)
