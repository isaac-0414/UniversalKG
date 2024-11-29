# Knowledge Graph Powered Question-Answering System
Combining Knowledge Graph and LLM to increase the factual accuracy of question-answering system

## Folder / File Descriptions
   * kg_save: saves the pickle format knowledge graph object
   * kg_visualization: saves HTML format knowledge graph visualization
   * vdb / subgraph_vdb: vector_database for knowledge graph
   * utils: All the helper functions
      * gpt.py: wrapper of GPT
      * kg_gen.py: All the functions for generating a knowledge graph
      * similarity.py: cosine similarity function
      * vdb.py: class and functions for managing a vector database
   * KnowledgeGraph.py: the knowledge graph class
   * main.py: entry point of constructing a knowledge graph
   * qa.py: entry point of using knowledge graph to answer a question
   * source: text being used to constructing a knowledge graph