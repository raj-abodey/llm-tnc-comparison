import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from prompts import TNCLegalPrompts_v2
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


def get_summary():
    output_file_name = "baseline_overall_os"

    # Read the input files
    jan_2015_tnc, mar_2023_tnc = read_tnc_files()

    llm = AzureChatOpenAI(
        azure_endpoint=api_base,
        api_version=api_version,
        deployment_name=deployment,
        api_key=api_key,
        temperature=0,
        seed=42,
    )

    prompt = TNCLegalPrompts_v2.system_prompt
    prompt = ChatPromptTemplate.from_template(prompt)

    partial_prompt = prompt.partial(document1=jan_2015_tnc, document2=mar_2023_tnc)

    chain = partial_prompt | llm

    user_query = "You are given two terms and condition documents,can you analyse both documents in depth and identify the key differences between them and summarize them? The summary should include more precise references to particular sections or clauses from both documents."
    response = chain.invoke({"user_input": user_query})
    print(response.content)
    output_file_name += ".txt"
    with open(f"./output/{output_file_name}", "w+") as f:
        f.writelines(response.content)


if __name__ == "__main__":
    get_summary()
