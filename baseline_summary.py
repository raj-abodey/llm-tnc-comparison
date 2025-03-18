import os
from typing import List

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from prompts import TNCLegalPrompts
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


def get_tnc_documents() -> List[Document]:
    """Returns the terms and conditions in the Document format

    Returns:
        List[Document]: List of documents ready to be consumed by downstream embedding or LLM models
    """

    jan_2015_tnc, mar_2023_tnc = read_tnc_files()

    doc_2015 = Document(page_content=jan_2015_tnc, metadata={"year": 2015})
    doc_2023 = Document(page_content=mar_2023_tnc, metadata={"year": 2023})

    documents = [doc_2015, doc_2023]
    return documents


def get_summaries(
    use_semantic_chunker=False, use_open_ai_embeddings=True, use_ollama_embedding=False
):
    output_file_name = "baseline_"
    if use_open_ai_embeddings:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=embedding_api_base,
            azure_deployment="abodey-embedding-model",
            openai_api_version=embedding_api_version,
        )
        output_file_name += "openaiEmb_"
    elif use_ollama_embedding:
        embeddings = OllamaEmbeddings(
            model="hf.co/TheBloke/law-LLM-GGUF:Q5_K_M",
        )
        output_file_name += "ollamaEmb_"

    else:
        model_name = "intfloat/multilingual-e5-large-instruct"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        output_file_name += "opensourceEmb_"
    # Read the input files
    jan_2015_tnc, mar_2023_tnc = read_tnc_files()
    if use_semantic_chunker:
        text_splitter = SemanticChunker(embeddings)
        texts = text_splitter.create_documents(
            [jan_2015_tnc, mar_2023_tnc], metadatas=[{"year": 2015}, {"year": 2023}]
        )
        output_file_name += "semanticChnk"
    else:
        doc_2015 = Document(page_content=jan_2015_tnc, metadata={"year": 2015})
        doc_2023 = Document(page_content=mar_2023_tnc, metadata={"year": 2023})

        documents = [doc_2015, doc_2023]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        output_file_name += "normalChnk"

    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})

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

    prompt = TNCLegalPrompts.system_prompt
    prompt = ChatPromptTemplate.from_template(prompt)

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    )

    user_query = "You are given a set of terms of conditions documents across two time periods, can you analyse both documents in depth and identify the key differences between them and summarize them. Highlight specific impact to the user. The summary should include more precise references to particular sections or clauses from both documents."
    response = retrieval_chain.invoke(user_query)
    print(response)
    output_file_name += ".txt"
    with open(f"./output/{output_file_name}", "w+") as f:
        f.writelines(response.content)


if __name__ == "__main__":
    get_summaries(
        use_open_ai_embeddings=True,
        use_semantic_chunker=False,
        use_ollama_embedding=False,
    )
    get_summaries(
        use_open_ai_embeddings=True,
        use_semantic_chunker=True,
        use_ollama_embedding=False,
    )
    get_summaries(
        use_open_ai_embeddings=False,
        use_semantic_chunker=False,
        use_ollama_embedding=False,
    )
    get_summaries(
        use_open_ai_embeddings=False,
        use_semantic_chunker=True,
        use_ollama_embedding=False,
    )
    get_summaries(
        use_open_ai_embeddings=False,
        use_semantic_chunker=False,
        use_ollama_embedding=True,
    )
    get_summaries(
        use_open_ai_embeddings=False,
        use_semantic_chunker=True,
        use_ollama_embedding=True,
    )
