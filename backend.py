import os
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from operator import itemgetter

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Database connection setup
def setup_database():
    db_user = os.getenv("AIVEN_USER")
    db_password = os.getenv("AIVEN_PASSWORD")
    db_host = os.getenv("AIVEN_HOST")
    db_port = os.getenv("AIVEN_PORT")
    db_name = os.getenv("AIVEN_DATABASE")
    
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}", schema=db_name, view_support=True)
    return db

# Load additional knowledge
def load_additional_knowledge():
    with open("additional_knowledge.txt", "r") as file:
        return file.read()

# Setup example selector
def setup_example_selector(examples):
    vectorstore = Chroma()
    vectorstore.delete_collection()
    return SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        vectorstore,
        k=2,
        input_keys=["input"],
    )

# Setup prompts
def setup_prompts(db, additional_knowledge, example_selector):
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}\nAccount ID: {accountId}\nSQL Query:"),
        ("ai", "{query}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        input_variables=["input", "top_k"],
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a MySQL expert. Given an input question and an account ID, create a syntactically correct MySQL query to run. Always include the account_id in your WHERE clause for filtering.\n\nHere is the relevant table info: {table_info}\n\nAdditional knowledge for query generation:\n{additional_knowledge}\n\nImportant notes:\n\n1. Always consider the currency when dealing with monetary values. The currency is stored in the 'currency' column of the 'ads' and 'campaigns' tables.\n2. When comparing or aggregating monetary values, ensure they are in the same currency or use appropriate conversion rates.\n3. Include the currency in your SELECT statement when querying monetary values.\nBelow are a number of examples of questions, account IDs, and their corresponding SQL queries."),
        few_shot_prompt,
        ("human", "{input}\nAccount ID: {accountId}"),
    ])

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

Context 
Conduct an in-depth analysis of the user's Meta ad campaigns, focusing on individual campaign performance, trends, and optimization opportunities. When information is limited or unclear, engage the user to gather more context. Follow these steps:
Data Collection and Campaign Identification:
a) Retrieve data for all active and recently concluded campaigns (past 90 days)
b) If data access is limited, ask the user:
"I can only access data for [X] campaigns. Are there specific campaigns you'd like me to focus on?"
c) Identify campaign objectives and types (e.g., conversions, traffic, awareness)
d) If campaign objectives are unclear, ask:
"Could you clarify the primary objectives for your main campaigns? This will help me provide more relevant insights."
Individual Campaign Performance Analysis:
a) For each campaign, analyze key metrics:
Return on Ad Spend (ROAS)
Cost Per Result (based on campaign objective)
Click-Through Rate (CTR)
Conversion Rate
Total Spend
Impressions
Reach
Frequency
Relevance Score or Quality Ranking
b) Compare each campaign's performance to:
Account averages
Previous period (e.g., last 30 days vs. 30 days before)
Industry benchmarks
c) Identify performance trends over time
d) If any metrics show unusual patterns, ask:
"I've noticed [Metric X] for [Campaign Y] is [unusually high/low]. Has there been any recent change in strategy for this campaign?"
Campaign Structure Evaluation:
a) Analyze the structure of each campaign (ad sets, ads)
b) Assess the alignment between campaign structure and objectives
c) Identify any structural issues that might be impacting performance
d) If the structure seems unusual, ask:
"Can you explain the reasoning behind the structure of [Campaign Z]? I want to ensure I'm interpreting it correctly."
Audience Analysis by Campaign:
a) Evaluate audience performance within each campaign
b) Identify best and worst-performing audiences
c) Analyze audience overlap between campaigns
d) If audience data is limited, ask:
"I have limited information about your audience targeting. Can you tell me about the target audiences for your key campaigns?"
Creative Performance by Campaign:
a) Analyze performance of different ad formats within each campaign
b) Identify top-performing ad creatives for each campaign
c) Evaluate ad copy effectiveness and messaging themes
d) If creative data is limited, ask:
"Could you describe the main creative approaches you're using in [Campaign X]? I'd like to provide more accurate insights on creative performance."
Budget and Bidding Analysis by Campaign:
a) Assess budget utilization and pacing for each campaign
b) Analyze the effectiveness of bidding strategies used
c) Identify opportunities for budget reallocation between campaigns
d) If budget information is unclear, ask:
"Can you provide more details about your budget allocation strategy across campaigns? Are there any spending constraints I should be aware of?"
Placement Performance by Campaign:
a) Evaluate ad performance across different placements for each campaign
b) Identify most and least effective placements per campaign
c) If placement data is limited, ask:
"Are you using specific placement strategies for different campaigns, or primarily automatic placements?"
Campaign-Specific Conversion Funnel Analysis:
a) Analyze the conversion path for each campaign
b) Identify drop-off points in the funnel
c) Compare funnel performance between campaigns
d) If conversion data is incomplete, ask:
"I'm seeing limited conversion data for [Campaign Y]. Can you describe the expected customer journey for this campaign?"
Cross-Campaign Insights:
a) Identify patterns or insights that emerge across multiple campaigns
b) Analyze how different campaigns might be impacting each other
c) Suggest opportunities for synergy between campaigns
Campaign Optimization Recommendations:
a) Provide 3-5 specific optimization recommendations for each campaign
b) Prioritize recommendations based on potential impact
c) Reference AdsNerd's knowledge base for best practices
d) After presenting recommendations, ask:
"Do these optimization suggestions align with your campaign goals? Are there any you'd like more details on?"
Competitive Analysis (if Ad Library data is available):
a) Compare campaign strategies to visible competitor approaches
b) Identify potential opportunities based on competitor activities
c) If competitor data is unavailable, ask:
"Are you aware of any specific campaign strategies your competitors are using that we should consider?"
Future Campaign Planning:
a) Based on the analysis, suggest ideas for future campaigns
b) Identify underutilized opportunities in the current campaign mix
c) Ask the user:
"Are there any new campaign ideas or objectives you're considering that we should factor into this analysis?"
Data Visualization:
a) Create clear, comparative charts for campaign performance
b) Visualize trends and patterns across campaigns
c) Ensure visualizations can be easily explained in text format
Summary and Next Steps:
a) Provide a concise summary of overall campaign performance
b) Highlight key areas for improvement across campaigns
c) Outline recommended next steps for campaign optimization
d) Suggest specific AdsNerd modules for deeper dives into areas of concern
e) After presenting the summary, ask:
"Based on this campaign analysis, which areas would you like to focus on improving first?"
Throughout the analysis, maintain a conversational tone and avoid jargon. Be prepared to explain any technical terms or concepts if the user requests clarification. Always specify the source of data, especially when referencing industry benchmarks or competitor insights.
If any critical data is missing or there are limitations in the analysis, clearly communicate this to the user and explain how it might impact the insights provided. Use the questions provided to gather more context and provide the most accurate and helpful analysis possible.
Remember, the goal is to provide the user with a clear, actionable understanding of their individual campaign performance and how campaigns work together, setting the stage for targeted optimizations and improved overall ad account performance.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer: """
    )

    return final_prompt, answer_prompt

# Setup chains
def setup_chains(db, llm, final_prompt, answer_prompt):
    generate_query_chain = create_sql_query_chain(llm, db, final_prompt)
    execute_query = QuerySQLDataBaseTool(db=db)
    rephrase_answer = answer_prompt | llm | StrOutputParser()

    return generate_query_chain, execute_query, rephrase_answer

# Main processing function
def process_question(question, account_id, generate_query_chain, execute_query, rephrase_answer, db, additional_knowledge):
    generated_query = generate_query_chain.invoke({
        "question": question,
        "accountId": account_id,
        "table_info": db.table_info,
        "additional_knowledge": additional_knowledge,
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

# Main execution
if __name__ == "__main__":
    db = setup_database()
    additional_knowledge = load_additional_knowledge()
    
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
        }
    ]
    
    example_selector = setup_example_selector(examples)
    
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    final_prompt, answer_prompt = setup_prompts(db, additional_knowledge, example_selector)
    
    generate_query_chain, execute_query, rephrase_answer = setup_chains(db, llm, final_prompt, answer_prompt)
    
    # Example usage
    question = "Which campaign has the most conversion rate?"
    account_id = "act_624496083171435"

    query, answer = process_question(question, account_id, generate_query_chain, execute_query, rephrase_answer, db, additional_knowledge)
    print("Generated SQL query:")
    print(query)
    print("\nInterpreted answer:")
    print(answer)


