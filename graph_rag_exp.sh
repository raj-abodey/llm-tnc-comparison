graphrag init --root ./graph_rag   

graphrag index --root ./graph_rag   

graphrag prompt-tune --root ./graph_rag --config ./graph_rag/settings.yaml --discover-entity-types

graphrag query \
--root ./graph_rag \
--method global \
--query "What are the key differences between 2015 and 2023 terms and conditions documents?Only use the context provided and no other additional information"|tail -n +5  > ./output/baseline_graphrag.txt

graphrag query \                 
--root ./graph_rag \
--method global \
--query "Based on the documents provided, can you standardise a common format for terms and conditions and explain what each section should contain?"|tail -n +5  > ./output/standardise_graphrag.txt