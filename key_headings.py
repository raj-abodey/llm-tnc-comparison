import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from extractor_schema import Response
from utils import read_tnc_files

# Define environment variables/API Key

# Embedding Model
embedding_api_base = os.getenv("AZURE_ENDPOINT")
embedding_api_version = "2024-02-01"
embedding_api_key = os.getenv("OPENAI_API_KEY")

# GPT 4-o config
api_base = os.getenv("GPT4_API_ENDPOINT")
api_version = os.getenv("GPT4_API_VERSION")
api_key = os.getenv("GPT4_API_KEY")
deployment = "gpt-4o"


jan_2015_tnc, mar_2023_tnc = read_tnc_files()

prompt = """You are a legal AI assisstant who has extensive knowledge about terms and conditions/terms of service. You will be given two terms and condition documents, 2015_document and 2023_document.Your task is to identify the key headings across both the documents that can then be used to summarise them into common topics.

    Answer the question based only on the following two contexts:
    2015_document:{document1}
    2023_document:{document2}
    
    """
llm = AzureChatOpenAI(
    azure_endpoint=api_base,
    api_version=api_version,
    deployment_name=deployment,
    api_key=api_key,
    temperature=0,
    seed=42,
)
prompt = ChatPromptTemplate.from_template(prompt)
llm_structured_output = llm.with_structured_output(Response)

partial_prompt = prompt.partial(document1=jan_2015_tnc, document2=mar_2023_tnc)
chain = partial_prompt | llm_structured_output

user_query = "You are given two terms and condition documents,can you analyse both documents in depth and identify the key subheadings that can be used to summarise the whole document."
response = chain.invoke({"user_input": user_query})
print(response.dict())
json_string = json.dumps(response.dict(), indent=4)
with open("./output/key_headings.json", "w") as f:
    print(json_string, file=f)
