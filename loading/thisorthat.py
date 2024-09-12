import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from operator import itemgetter
from langchain.tools.base import BaseTool
import csv

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Get database credentials from environment variables
db_user = os.getenv("AIVEN_USER")
db_password = os.getenv("AIVEN_PASSWORD")
db_host = os.getenv("AIVEN_HOST")
db_port = os.getenv("AIVEN_PORT")
db_name = os.getenv("AIVEN_DATABASE")

# db_user = "root"
# db_password = "password123#"
# db_host = "localhost"
# db_port = "3306"
# db_name = "adsnerd"

# Create database connection
db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
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

#

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
        AND ci.date_start >= UNIX_TIMESTAMP(DATE_SUB(CONVERT_TZ(NOW(), '+00:00', '+05:30'), INTERVAL 30 DAY))
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
AND asi.date_start >= UNIX_TIMESTAMP(DATE_SUB(CONVERT_TZ(NOW(), '+00:00', '+05:30'), INTERVAL 30 DAY))
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
    AND ci.date_start >= UNIX_TIMESTAMP(DATE_SUB(CONVERT_TZ(NOW(), '+00:00', '+05:30'), INTERVAL 30 DAY))
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
    AND ci.date_start >= UNIX_TIMESTAMP(DATE_SUB(CONVERT_TZ(NOW(), '+00:00', '+05:30'), INTERVAL 30 DAY))
GROUP BY c.objective
            ORDER BY total_spend DESC;
        """,
    },
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

# loader = CSVLoader(file_path="new-sample-schema.csv" )
# table_info = loader.load()

# # Create a prompt template for the table info
# table_info_template = PromptTemplate(
#     input_variables=[
#         "table",
#         "column",
#         "data_type",
#         "description",
#         "is_primary_key",
#         "foreign_key",
#         "reference"
#     ],
#     template="{table}.{column} ({data_type}): {description}. Primary Key: {is_primary_key}. Foreign Key: {foreign_key} Reference: {reference}",
# )

# # Format the table info
# formatted_table_info = "\n".join(
#     [
#         table_info_template.format(
#             table=doc.metadata["Table Name"],
#             column=doc.metadata["Column Name"],
#             data_type=doc.metadata["Data Type"],
#             description=doc.metadata["Description"],
#             is_primary_key=doc.metadata["Is Primary Key"],
#             foreign_key=doc.metadata["Foreign Key Reference"] or "None",

#         )
#         for doc in table_info
#     ]
# )


def load_schema_from_csv(file_path):
    schema = {}
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames
        print(headers)

        # Check for alternative column names
        table_col = next((col for col in headers if "table" in col.lower()), None)
        column_col = next((col for col in headers if "column" in col.lower()), None)
        data_type_col = next(
            (col for col in headers if "data type" in col.lower()), None
        )
        description_col = next(
            (col for col in headers if "description" in col.lower()), None
        )
        primary_key_col = next(
            (col for col in headers if "primary key" in col.lower()), None
        )
        foreign_key_col = next(
            (col for col in headers if "foreign key" in col.lower()), None
        )

        if not all(
            [
                table_col,
                column_col,
                data_type_col,
                description_col,
                primary_key_col,
                foreign_key_col,
            ]
        ):
            raise ValueError("CSV file is missing required columns")

        for row in reader:
            table_name = row[table_col]
            if table_name not in schema:
                schema[table_name] = []
            schema[table_name].append(
                {
                    "Column Name": row[column_col],
                    "Data Type": row[data_type_col],
                    "Description": row[description_col],
                    "Is Primary Key": row[primary_key_col],
                    "Foreign Key Reference": row[foreign_key_col],
                }
            )
    return schema


# Usage
file_path = r"C:\Sreerag\gen-ai\loading\sample.csv"
schema = load_schema_from_csv(file_path)


def format_schema_for_prompt(schema):
    formatted_schema = ""
    for table_name, columns in schema.items():
        formatted_schema += f"Table: {table_name}\n"
        for column in columns:
            formatted_schema += f"  - {column['Column Name']} ({column['Data Type']}): {column['Description']}"
            if column["Is Primary Key"] == "Yes":
                formatted_schema += " (Primary Key)"
            if column["Foreign Key Reference"]:
                formatted_schema += f" (Foreign Key: {column['Foreign Key Reference']})"
            formatted_schema += "\n"
        formatted_schema += "\n"
    return formatted_schema


# Usage
formatted_schema = format_schema_for_prompt(schema)

# user query to embedding -> search for similar examples -> use the context to generate query

# similarity search for docs

# csv files for  storing the db schema


final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a MySQL expert. Given an input question, account ID, and additional context, create a syntactically correct MySQL query to run.
              Always include the account_id in your WHERE clause for filtering.

Important notes:
1. Always consider the currency when dealing with monetary values. The currency is stored in the 'currency' column of the 'ads' and 'campaigns' tables.
2. When comparing or aggregating monetary values, ensure they are in the same currency or use appropriate conversion rates.
3. Include the currency in your SELECT statement when querying monetary values.
4.Do not use ad_set_key in grouping since we will have multiple ad_set_key for a single adset instead use name
5.Always change the timestamp to IST Timezone by using this UNIX_TIMESTAMP(DATE_SUB(CONVERT_TZ(NOW(), '+00:00', '+05:30')))
Below are a number of examples of questions, account IDs, and their corresponding SQL queries
Here is the relevant table info and relationships: 
{table_info}

.""",
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
        generated_query = generate_query.invoke(
            {
                "question": question,
                "accountId": account_id,
                "table_info": formatted_schema,
            }
        )
        clean_query = (
            generated_query.strip().replace("```sql", "").replace("```", "").strip()
        )

        try:
            result = execute_query.invoke(clean_query)
            answer = rephrase_answer.invoke(
                {
                    "question": question,
                    "query": clean_query,
                    "result": result,
                }
            )
            return clean_query, answer
        except Exception as e:
            return clean_query, f"An error occurred: {str(e)}"
    elif decision == "USE_LLM":
        answer = answer_general_question(question)
        return None, answer
    else:
        return (
            None,
            "Unable to determine how to process the question. Please try rephrasing your query.",
        )


# Example usage
if __name__ == "__main__":
    questions = [
        # "Adset with highest impressions in b/w 1st september 2024 and 10th september 2024",
        "Adset with lowest impressions in b/w 1st september 2024 and 10th september 2024",

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
