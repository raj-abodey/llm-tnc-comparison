# llm-tnc-comparison
Compare terms and conditions documents using LLMs

Objective: 
1. Evaluate the use of LLM for comparing changes in terms and conditions documents.
2. Standardised format for T&C 

# Approach

Different solutions were experimented. This included evaluating different embedding models and open source models. The main LLM used is Azure GPT4-o. 

Embedding models evaluated:
1. Openai Ada embedding model (ref: openaiEMb)
2. Huggingface "multilingual-e5-large-instruct" (Selected based on MTEB) (ref: opensourceEmb)
3. Huggingface legal domain quantised model: "law-LLM-GGUF:Q5_K_M" (ref: ollamaEmb)

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

All the results can be found in the ./output folder.

## Overall 
Given the resource constraints, only gpt4-o was used as the main LLM. Different embedding models were investigated. Below we summarise the results based on llm-as-a-judge and g-eval metrics.

1. The best overall approach was when parsing the full documents as a context. This is the baseline to compare against.
2. The next best approach, which is more comprehensive is to combine an extraction pipeline with a summarisation chain. We can identify key topics and iteratively call extraction-summariser pipeline. This is more comprehensive as we identify the key differences within individual sub-topics.
3. Graphrag seems to outperform any traditional RAG, as extraction of knowledge graph enables better information retreival. 
4. In terms of embedding (evaluated only for RAG solutions), open source and domain specific quantised model outperform commercial solution (openai-ada). 
5. We also tried to evaluate semantic chunking with normal chunking. In this particular case, normal chunking seems to perform better than semantic chunking. 


# Standardisation of T&C documents

Two different approaches to identify common topics for standardising T&C:
1. LLM with structured output: Using prompt engineering and structured output, we can ideally look at ways to understand standard components. (see /output/key_headings.json)
2. Using Graphrag: This approach essentially builds a knowledge graph which captures the underlying structure of the document. We can then query the graphrag to extract common structures.(see /output/standardise_graphrag.txt)

Looking at the results, graphrag approach seems to generalise the structure well. LLM with structured output identifies common thread across both the documents, however, seems to be very specific to apple.

# To replicate results

## Directories
1. data: Contains the input files
2. output: Contains all the output from the llm 

## Basline with no RAG
 ```python 
 python baseline_summary_no_rag.py
 ```

## Basline
 ```python 
 python baseline_summary.py
 ```
## Graphrag
```sh
sh graph_rag_exp.sh
```

## Standardisation using structured output
```python
python key_headings.py
```

## Extractive Summary
```python
python extractive_summary.py
```

# Conclusion, next steps and future direction

To conclude, gpt-4o along with the whole documents as a context is the best approach. However, not in all cases we will be able to parse the whole documents into the prompt as context window will be limited. We can use extractive-summariser pipeline or use graphrag. For standardising t&c documents, graphrag provides the best approach. We have evaluated these using two approaches based on using LLMs. 

## Next steps
1. We can extend the extractor-summariser pipeline into an agentic approach. Try multi-agentic approach.
2. Domain specific embedding seem to improve performance. We can investigate the use of the domain specific fine-tuned LLM as a replacement to gpt-4o.
3. Prompt optimisation using DsPy
4. Evaluation of llm output is still an active research field. We used LLMs based approach, however, these approaches will have bias and high computation cost, and inconsistent across different summaries. Ideally, we would need human in the feedback loop. This can then be augmented to fine-tune LLMs. Frameworks such as flow judge or uptrain consists different evaluation metrics which can be used to evaluate LLM output.

