from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import shantanu_snow
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

#Mistral
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.5)

#Personal_Mistral
template = """
Following is the context, based on the context answer given question:
Context={context}
{question}"""
prompt = PromptTemplate.from_template(template)
prompt_final=PromptTemplate(template=template, input_variables=['question', 'Context'])
llm_chain = LLMChain(prompt=prompt_final, llm=llm)

def ask_mistral(question):      
    return llm(question)

def personal_mistral(question, db):
    context=db.similarity_search(query=question ,fetch_k=4)
    return llm_chain.run(question=question, context=context)

def personal_mistral_snowflake(question, db):
    result=[]
    context=db.similarity_search(query=question ,fetch_k=4)
    result = llm_chain.run(question=question, context=context)
    start_index = result.find("`") + 3  # Add 3 to skip the starting "`" and space
    end_index = result.rfind("```")  # Use rfind to get the last occurrence
    sql_code = result[start_index:end_index]
    sql_code='--'+sql_code
    print("----")
    print(sql_code)
    print("----")
    l1=sql_code.split(";")
    for query in l1:
        print(query)
        result.append(shantanu_snow.snowflake_run(query))
    return str(result)

def mistral_csv(df, question):
    df_agent = create_pandas_dataframe_agent(llm, df)
    def handle_parsing_error(error):
        print(f"Error parsing LLM output: {error}")
    response = df_agent.invoke(question, handle_parsing_errors=handle_parsing_error)
    return response['output']