# llm-tnc-comparison
Compare terms and conditions documents using LLMs

Objective: 
1. Evaluate the use of LLM for comparing changes in terms and conditions documents.
2. Standardised format for T&C 

# Approach

Different solutions were experimented. This included evaluating different embedding models and open source models. The main LLM used is Azure GPT4-o. 

Embedding models evaluated:
1. Openai Ada embedding model
2. Huggingface "multilingual-e5-large-instruct" (Selected based on MTEB)
3. Huggingface legal domain quantised model: "law-LLM-GGUF:Q5_K_M"

## Different approaches for document comparison and summarisation
1. Baseline with no RAG: Parse the documents directly in prompts and query for differences
2. Baseline with RAG: Instead of parsing the documents directly, we use a RAG solution. Here we evaluate different embedding models and different chunking techniques.
3. Baseline with graphrag: We use graphRAG, where we build knowledge graph on the documents, which then allows us to query efficiently.
4. Extractive Summary: In order to get all the relevant information, we first identify key topics. Then we first use LLM to extract the relevant piece of information from each document, which is then passed on to a different summarising chain to comment on the differences. This approach allows us to focus on key segments and make sure tha relevant information is passed on. Such an approach will allow us to query efficiently with a small context window.
5. Agentic solution: Another potential approach is to combine graphrag with an agentic approach (ReAct or multi-agent). This will allow us to extract the relevant information and summarise, i.e automatically carry out extractive summary by agent exploring or using tools.

## Evaluation of the document comparison

Two types of approach was used to evaluate the performance. Both the approaches were based LLMs.
1. LLM-as-a-judge type: In this approach, we design a LLM chain to be the judge and ask the llm to rank the llm output.
2. G-eval: This is also similar to the above, but uses chain of thought and is customisable to different tasks. 

# Results

## Overall 


# Standardisation of T&C documents

Two different approaches to identify common topics for standardising T&C:
1. LLM with structured output: Using prompt engineering and structured output, we can ideally look at ways to understand standard components.
2. Using Graphrag: This approach essentially builds a knowledge graph which captures the underlying structure of the document. We can then query the graphrag to extract common structures.(see /output/standardise_graphrag.txt)

# Next steps and future direction

## G-Eval

## Other metrics 

ROGUE, BertScore, BLEU can be used to evaluate LLM output. Each of them have their own advantages and disadvantages.