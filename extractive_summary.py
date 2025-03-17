import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from utils import read_tnc_files

# GPT 4-o config
api_base = os.getenv("GPT4_API_ENDPOINT")
api_version = os.getenv("GPT4_API_VERSION")
api_key = os.getenv("GPT4_API_KEY")
deployment = "gpt-4o"


def get_extractive_summary():
    """Extracts individual topics from both documents and then asks LLM to summarise the difference"""
    key_headings = {
        "Introduction": "Introduces the purpose of the document, services or products covered, scope and entity responsible",
        "Privacy and Data protection": "Explains how user data is collected, used, and protected, in accordance with the company's privacy policy. It should also mention any third-party involvement in data processing and the user's rights regarding their personal data",
        "Licensing and Intellectual Property": "Details regarding the licensing terms for using the services or products, including any end-user license agreements (EULAs) and intellectual property rights.",
        "User Obligations": "outline of the responsibilities and obligations of the user, including compliance with usage rules, legal requirements, and any specific conditions related to the use of the services or products.",
    }
    jan_2015_tnc, mar_2023_tnc = read_tnc_files()
    llm = AzureChatOpenAI(
        azure_endpoint=api_base,
        api_version=api_version,
        deployment_name=deployment,
        api_key=api_key,
        temperature=0,
        seed=42,
    )
    extraction_prompt = """ You are an expert extraction algorithm. You will be given a topic and a description, your task is to extract the relevant information from the context provided. If you do not know the value of the attribute asked to extract, return null.
    topic:{topic}
    description: {description}
    context:{context}
    """
    extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt)

    descriptive_prompt = """You are an expert legal analyser for terms and conditions. You will be given a topic and two documents, document1_2015 and document2_2023. Your task is to compare the two documents and summarise the key differences. Highlight the impact and significance of the changes.
    topic:{topic}
    document1_2015:{document1}
    document2_2023:{document2}
    """
    descriptive_prompt = ChatPromptTemplate.from_template(descriptive_prompt)

    final_summary = ""

    for key in key_headings:
        partial_prompt1 = extraction_prompt.partial(
            topic=key, description=key_headings[key], context=jan_2015_tnc
        )
        partial_prompt2 = extraction_prompt.partial(
            topic=key, description=key_headings[key], context=mar_2023_tnc
        )
        extraction_chain1 = partial_prompt1 | llm
        extraction_chain2 = partial_prompt2 | llm

        user_query = " Can you extract the relevant section as it is described?"
        extraction_doc1 = extraction_chain1.invoke({"user_input": user_query})
        extraction_doc2 = extraction_chain2.invoke({"user_input": user_query})
        # print(extraction_doc1.content)
        # print("-----------")
        # print(extraction_doc2.content)
        summariser_partial_prompt1 = descriptive_prompt.partial(
            topic=key,
            document1=extraction_doc1.content,
            document2=extraction_doc2.content,
        )
        summariser_chain = summariser_partial_prompt1 | llm
        user_query = "Can you identify the key differences between the two documents? "
        summary = summariser_chain.invoke({"user_input": user_query})
        print(summary.content)
        print(10 * "**")
        final_summary += summary.content
    output_file_name = "extractive_summary.txt"
    with open(f"./output/{output_file_name}", "w+") as f:
        f.writelines(final_summary)


if __name__ == "__main__":
    get_extractive_summary()
