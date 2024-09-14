from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.prompts import ChatPromptTemplate
import os
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.tools.base import BaseTool
from operator import itemgetter
from typing import List, Dict

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI models
embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

# Initialize SQL database connection
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
vector_store = Chroma(persist_directory="./chunk", embedding_function=embedding_model)

# Define prompts
text2sql_prompt = ChatPromptTemplate.from_template(
    """
You are an expert in converting natural language to SQL queries for meta ads analysis.
Use the following context to help formulate the SQL query:
{context}

User query: {query}

Important notes:

1. Always consider the currency when dealing with monetary values. The currency is stored in the 'currency' column of the 'ads' and 'campaigns' tables. Use it when monetary values are considered in any query.
2. When comparing or aggregating monetary values, ensure they are in the same currency or use appropriate conversion rates.
3. Include the currency in your SELECT statement when querying monetary values.
4. Always include the name of campaigns, ad sets, and ads in your SELECT statement, not just their IDs.
5. Use appropriate JOIN statements to retrieve names from related tables (e.g., campaigns, ad_sets, ads).

Generate a SQL query to answer the user's question:
"""
)

final_answer_prompt = ChatPromptTemplate.from_template(
    """
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
- Always refer to campaigns, ad sets, and ads by their names, not their IDs.
- Maintain a professional yet conversational tone.
- Prioritize accuracy, relevance, and actionable insights.
- Use Meta-specific terminology where appropriate (e.g., ad sets, campaigns, placements).
- Consider the broader context of the Meta ads ecosystem (e.g., algorithm learning, audience saturation).
- If the data spans a time period, note any temporal trends or changes.

User query: {query}
SQL result: {sql_result}

Analysis and Insights:
"""
)


# Define custom tool
class KnowledgeBaseTool(BaseTool):
    name = "Knowledge Base"
    description = "Use this tool to query the knowledge base for relevant context"

    def _run(self, query: str) -> str:
        results = vector_store.similarity_search(query, k=2)
        print(results)
        return "\n".join([doc.page_content for doc in results])

    async def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


# Create SQL agent
sql_agent = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True)


# Define the main function to run the agent
def analyze_meta_ads(query: str) -> Dict[str, str]:
    # Step 1: Query knowledge base for context
    kb_tool = KnowledgeBaseTool()
    kb_context = kb_tool._run(query)

    # Step 2: Generate and execute SQL query
    sql_response = sql_agent.invoke(
        {
            "input": f"""
Generate and execute a SQL query to answer: {query}

Make sure to:
1. Include names in the query, not just their IDs.
2. Use appropriate JOIN statements and table relationships as described in the context.
3. Always consider the currency when dealing with monetary values. The currency is stored in the 'currency' column of the 'ads' and 'campaigns' tables.
4. When comparing or aggregating monetary values, ensure they are in the same currency or use appropriate conversion rates.
5. Include the currency in your SELECT statement when querying monetary values.

Use the following context from the knowledge base to know what tables to query and what metrics to use:
{kb_context}
"""
        }
    )
    sql_result = sql_response["output"]

    # Step 3: Generate final answer with insights
    final_answer_chain = (
        {
            "query": itemgetter("query"),
            "sql_result": itemgetter("sql_result"),
            "context": itemgetter("context"),
        }
        | final_answer_prompt
        | llm
        | StrOutputParser()
    )

    final_answer = final_answer_chain.invoke(
        {"query": query, "sql_result": sql_result, "context": kb_context}
    )

    return {"sql_query": sql_result, "final_answer": final_answer}


# Example usage
user_query = "Which creative got the most unique clicks at the lowest cost?"
result = analyze_meta_ads(user_query)
print("SQL Query:", result["sql_query"])
print("\nFinal Answer:", result["final_answer"])
