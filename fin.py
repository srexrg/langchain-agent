import os
from typing import List
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter
import pandas as pd
import csv
from examples import examples

# Load environment variables
load_dotenv()

# Database connection setup
db_user = os.getenv("AIVEN_USER")
db_password = os.getenv("AIVEN_PASSWORD")
db_host = os.getenv("AIVEN_HOST")
db_port = os.getenv("AIVEN_PORT")
db_name = os.getenv("AIVEN_DATABASE")

# Initialize database connection
db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
    schema=db_name,
    view_support=True,
)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Initialize LLM


# Replace instances of ChatOpenAI(model="gpt-4", temperature=0) with claude_llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load schema from CSV
def load_schema_from_csv(file_path):
    schema = {}
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames

        required_columns = {
            "table": "table",
            "column": "column",
            "data type": "data type",
            "description": "description",
            "primary key": "primary key",
            "foreign key": "foreign key",
        }

        column_mapping = {}
        for required, alternatives in required_columns.items():
            column_mapping[required] = next(
                (col for col in headers if alternatives in col.lower()), None
            )

        if not all(column_mapping.values()):
            raise ValueError("CSV file is missing required columns")

        for row in reader:
            table_name = row[column_mapping["table"]]
            if table_name not in schema:
                schema[table_name] = []
            schema[table_name].append(
                {
                    "Column Name": row[column_mapping["column"]],
                    "Data Type": row[column_mapping["data type"]],
                    "Description": row[column_mapping["description"]],
                    "Is Primary Key": row[column_mapping["primary key"]],
                    "Foreign Key Reference": row[column_mapping["foreign key"]],
                }
            )
    return schema


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


# Load schema
file_path = r"C:\Sreerag\gen-ai\loading\sample.csv"
schema = load_schema_from_csv(file_path)
table_details = format_schema_for_prompt(schema)


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


# Define a wrapper model for the list of tables
class TableList(BaseModel):
    """List of tables in SQL database."""

    tables: List[Table] = Field(description="List of relevant tables in SQL database.")


table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""


def get_tables(question):
    prompt = PromptTemplate(
        input_variables=["table_details", "question"],
        template="{table_details}\n\nQuestion: {question}\n\nRelevant tables:",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(table_details=table_details_prompt, question=question)
    # Parse the response to extract table names
    return [table.strip() for table in response.split(",")]


select_table = RunnablePassthrough.assign(
    structured_output=lambda x: llm.with_structured_output(TableList).invoke(
        {"input": x["question"], "system_message": table_details_prompt}
    )
) | (lambda x: get_tables(x["structured_output"]))

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
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

# Set up final prompt
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specified.
         Important notes:
1. Always consider the currency when dealing with monetary values.
2. When comparing or aggregating monetary values, ensure they are in the same currency or use appropriate conversion rates.
3. Include the currency in your SELECT statement when querying monetary values.
4. Do not use ad_set_key in grouping since we will have multiple ad_set_key for a single adset instead use name
5. Always change the timestamp to IST Timezone by using this UNIX_TIMESTAMP(DATE_SUB(CONVERT_TZ(NOW(), '+00:00', '+05:30')))
6. Always include a WHERE clause to filter results for the given account ID: account_id = {accountId}
Below are a number of examples of questions, account IDs, and their corresponding SQL queries


Here is the relevant table info: {table_info}

Below are a number of examples of questions and their corresponding SQL queries. These examples are just for reference and should be considered while answering follow-up questions""",
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Question: {input} Account Id: {accountId}"),
    ]
)

# Set up query generation and execution

generate_query = create_sql_query_chain(llm, db, final_prompt)
execute_query = QuerySQLDataBaseTool(db=db)


# Set up answer generation
answer_prompt = PromptTemplate.from_template(
    """You are an expert analyst for Meta (Facebook) advertising. Analyze the provided data and answer the user's question. Follow these guidelines:

1. Data Interpretation:
   - Analyze the SQL query results in the context of the user's question.
   - If data is limited or unclear, note this in your response.

2. Performance Analysis:
   - Focus on key metrics: ROAS, CPA, CTR, Conversion Rate, Ad Spend, Impressions, Reach.
   - Compare current performance to previous periods and industry benchmarks (if available).
   - Identify top and bottom performing elements (campaigns, ad sets, ads).

3. Insights and Recommendations:
   - Provide 3-5 key insights based on the data.
   - Offer actionable recommendations for improvement.
   - Suggest areas that might need deeper analysis.

4. Clear Communication:
   - Use a professional yet conversational tone.
   - Explain technical terms if necessary.
   - Present information in a logical, easy-to-understand manner.

5. Data Transparency:
   - Clearly state the time period of the data.
   - Mention any limitations in the data or analysis.

6. Follow-up:
   - Suggest 1-2 follow-up questions or areas for further investigation.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Provide your analysis and answer below:
"""
)

rephrase_answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | execute_query
    )
    | rephrase_answer
)

# Set up query or knowledge decision chain
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


# Set up chat history
history = ChatMessageHistory()


# Main processing function
def process_question(question,account_id):
    decision = query_or_knowledge_chain.invoke({"question": question})
    print(decision)

    if decision == "QUERY_DB":

        generated_query = generate_query.invoke(
            {
                "question": question,
                "messages": history.messages,
                "accountId": account_id,
            }
        )

        clean_query = (
            generated_query.strip().replace("```sql", "").replace("```", "").strip()
        )
        print(clean_query)

        try:
            result = execute_query.invoke(clean_query)
            print(result)
            answer = rephrase_answer.invoke(
                {
                    "question": question,
                    "query": clean_query,
                    "result": result,
                }
            )

            history.add_user_message(question)
            history.add_ai_message(answer)

            return clean_query, answer
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            history.add_user_message(question)
            history.add_ai_message(error_message)
            return clean_query, error_message
    elif decision == "USE_LLM":
        answer = answer_general_question(question)
        history.add_user_message(question)
        history.add_ai_message(answer)
        return None, answer
    else:
        error_message = "Unable to determine how to process the question. Please try rephrasing your query."
        history.add_user_message(question)
        history.add_ai_message(error_message)
        return None, error_message


# Example usage
if __name__ == "__main__":
    question = "Best performing campaign in the last month?"
    account_id = "act_624496083171435"
    query, answer = process_question(question, account_id)
    if query:


        print("Generated SQL query:")
        print(query)
    print("\nAnswer:")
    print(answer)
