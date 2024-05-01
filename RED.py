import os
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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”

# Initialize HuggingFacePipeline with model ID and task
hf = HuggingFacePipeline.from_model_id(
    model_id="Snowflake/snowflake-arctic-instruct”,
    task="text-generation"
)

# Define prompt template
template = """Following is the question please answer it:
{question}
"""
prompt = PromptTemplate.from_template(template)

# Compose pipeline with prompt and HuggingFace pipeline
chain = prompt | hf

# Define the question
question = "Who is the prime minister of India ?"

# Invoke the pipeline and print the response
print(chain.invoke({"question": question}))