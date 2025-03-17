import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from prompts import EvaluatorPrompts_v2
from utils import read_tnc_files

# Embedding Model
embedding_api_base = os.getenv("AZURE_ENDPOINT")
embedding_api_version = "2024-02-01"
embedding_api_key = os.getenv("OPENAI_API_KEY")

# GPT 4-o config
api_base = os.getenv("GPT4_API_ENDPOINT")
api_version = os.getenv("GPT4_API_VERSION")
api_key = os.getenv("GPT4_API_KEY")
deployment = "gpt-4o"


def evaluate_llm_output(file_name: str) -> None:
    jan_2015_tnc, mar_2023_tnc = read_tnc_files()

    llm = AzureChatOpenAI(
        azure_endpoint=api_base,
        api_version=api_version,
        deployment_name=deployment,
        api_key=api_key,
        temperature=0,
        frequency_penalty=1,
        presence_penalty=0,
        seed=42,
    )
    prompt = EvaluatorPrompts_v2.system_prompt
    prompt = ChatPromptTemplate.from_template(prompt)

    with open(file_name, "r") as file:
        llm_output = file.read()
    print(llm_output)
    partial_prompt = prompt.partial(
        summary=llm_output, document1=jan_2015_tnc, document2=mar_2023_tnc
    )
    chain = partial_prompt | llm

    user_query = "You are given a summary of differences between two terms and condition documents, can you evaluate the quality of the report. Provide instructions on how to improve the score"
    response = chain.invoke({"user_input": user_query})
    print(response.content)


if __name__ == "__main__":
    file_name = "./output/baseline_graphrag.txt"
    file_name = "./output/baseline_openaiEmb_normalChnk.txt"
    file_name = "./output/baseline_openaiEmb_semanticChnk.txt"
    file_name = "./output/baseline_opensourceEmb_normalChnk.txt"
    file_name = "./output/baseline_opensourceEmb_semanticChnk.txt"
    file_name = "./output/baseline_overall.txt"
    evaluate_llm_output(file_name)
