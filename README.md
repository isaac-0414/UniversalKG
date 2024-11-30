# Universal Knowledge Graph and Question-Answering System Powered By It

![License](https://img.shields.io/github/license/isaac-0414/auto-hacker)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## Table of Contents

- [Overview](#overview)
- [Method](#method)
  - [UKG Generation](#ukg-generation)
  - [UKG Querying](#ukg-querying)
  - [UKG Ontology(Type)](#ukg-ontologytype)
- [Experiment](#experiment)
- [Folder / File Descriptions](#folder--file-descriptions)
- [Usage](#usage)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

The Universal Knowledge Graph (UKG) is my ongoing research project endeavors to establish a standardized framework for constructing and querying knowledge graphs, ultimately enabling a more structured and efficient representation of information extracted from textual data. With the ever-growing volume of unstructured information available, UKG aims to provide a systematic approach to categorizing data into entities, relationships, and attributes, creating an interconnected web that mirrors the nature of real-world information. (For example, in the sentence "Isaac Zheng is an undergrad at UIUC from 2021 to 2025", my KG would have "Isaac Zheng" and "UIUC" as entities, "be_an_undergrad_at" as the relation, and "start_time: 2021" and "end_time: 2025" as the attributes).

By leveraging advanced Large Language Model (LLM), UKG facilitates the seamless transformation of input text into a coherent knowledge graph. I also experimented with incorporating types(ontologies) to enhance the semantic depth and consistency of each node and relationship, ensuring an accurate and context-aware categorization. Additionally, UKG employs a hybrid method combining textual and node embeddings to optimize the accuracy and relevance of query results.

The Universal Knowledge Graph project aims to provide the tools necessary to uncover hidden insights and foster a deeper understanding of textual information through a unified and comprehensive graphical framework.

## Method

The methodology employed in the Universal Knowledge Graph (UKG) project revolves around two main processes: the generation of knowledge graphs from textual data and the querying of these graphs to retrieve pertinent information. This section outlines the strategies used to systematically transform text into a structured graph and the techniques for accurately querying that graph. This section also includes generation of an ontology system corresponding to the UKG.

### UKG Generation

The UKG generation process involves systematically breaking down textual data into its fundamental components - entities, relationships, and attributes. By employing the following steps, it effectively constructs a comprehensive knowledge graph from the input text.

1. **Entity Extraction**: Extract all the entities in the given text chunk as well as attributes/properties related to it, since the entities that GPT extract are sometimes attributes/properties, and it is difficult to prevent that, I also added a validation process.
2. **Phrase Selection**: Extract all the sentences that contains a given entity.
3. **Entity Disambiguation**: Given a chunk of text, replace all the pronouns, abbreviations, acronyms, and last names with the names to which they refer to in the context of the paragraph.
4. **Relations Extraction**: Extract all the triples of [head_entity, relation, tail_entity] in the given text chunk, as well as attributes/properties corresponding to that relation, in this step, we will iterate over all the entities we previously got from **Entity Extraction**, and in each iteration, we will do the following:
   1) **Phrase Selection**: Same as above, extract all the sentences that contains the entity.
   2) **Mention Recognition**: Given a list of entities and the sentences we get from last substep, extract all the entities in this list that occurred in the sentences.
   3) **Relation Extraction around a single entity**: With a target entity and all the entities it might has relation with, we will extract the relationship between them and attributes relate to that property.

### UKG Querying

The querying framework of UKG is designed to deliver precise information retrieval from the constructed knowledge graph through a categorized question approach. By differentiating between basic and advanced questions and using specialized techniques for entity, relationship, and attribute inquiries, UKG ensures efficient and context-aware responses.

1. **Question Classification**: I will first classify the incoming question into 2 categories, whether it is a "Basic Question" or an "Advanced Question".
   1) **Basic Question**: A basic question is a question about ONE entity, relationship, or attribute. I will further classify the quesiton into an "Entity Quesiton", "Relationship Question" and "Attribute Question".
   2) **Advanced Quesiton**: An advanced question can be broken down into multiple Basic Question. I will process each Basic Question and combine the result together into the final result.
2. **Entity Question**: Below is an example how I deal with an entity question, where I first rephrase the question to use "[ENTITY]" to replace the Entity of interest, and process this rephrased question into a small UKG, and collect all the other entites in this small UKG, as well as their path to the target Entity. Then we access the large UKG the serves as the knowledge base, locating each of those "other entities" (with both direct match and texual+node embedding), and do a DFS pattern path matching to go from those entities to a candidate target entity following the path we found in the small UKG. Normally, after the process there should be only one candidate entity, if there is more than one, I will find use the one from the path that has the higher cosine similarity with the path in small UKG.<br/>
<p align="center"><img src="/images/UKG-entity.png" width=70%, alt="Entity question example"></p>

3. **Relationship Question**: A relationship should only contain two entities and asks about the relationship between them. So we just find those two entities, locate them in the larger KG, and get information about relationship between them, an example of this type of question shown below.<br/>
<p align="center"><img src="/images/UKG-relation.png" width=30%, alt="Relationship question example"></p>

5. **Attribute Question**: This type of question can be complicated, currently I only explored the two most straightforward type of it, as shown below in this example.<br/>
<p align="center"><img src="/images/UKG-attribute.png" width=70%, alt="Attributes question example"></p>

### UKG Ontology(Type)
During the generation of the Universal Knowledge Graph, an ontology(type) system is also generated to categorize and label each entity with a specific type. This ontological framework resembles the class structure in computer science. For instance, "BMW" is categorized under the "Car Manufacturer" class, which is a subset of the "Organization" class, ultimately inheriting from the "Thing" class. Although the current implementation is limited to generating a knowledge graph from a single text segment, future enhancements aim to enable scalability. This would allow for the expansion of both the knowledge graph and its concomitant ontology system.

## Experiment
I did this research project in my spare time, apart from my commitments to my startup, lab work, and school work. Due to the lack of funding and extra effort, I only did some small-scale experiments, and result turned out to be satisfactory, you can find two of them in the `/examples` folder.

## Folder / File Descriptions
    .
    ├── /kg_save                      # saves the pickle format knowledge graph object
    ├── /kg_visualization             # saves HTML format knowledge graph visualization
    ├── /vdb                          # vector_database for knowledge graph.
    ├── /subgraph_vdb                 # vector_database for small UKG.
    ├── /examples                     # Contains the sourcetext I used to test UKG generation.
    ├── /utils                        # All the helper functions
    │   ├── gpt.py                    # wrapper of GPT functionalities
    │   ├── kg_gen.py                 # All the functions for generating a UKG
    │   ├── similarity.py             # cosine similarity function
    │   ├── vdb.py                    # class and functions for managing a vector database
    ├── KnowledgeGraph.py             # the UKG class
    ├── main.py                       # entry point of constructing a knowledge graph
    ├── qa.py                         # entry point of using knowledge graph to answer a question


## Installation

### Requirements

- Python 3.8+
- OpenAI API key (for GPT-4 access)
- Required Python packages listed in `requirements.txt` (Can be installed with provided shell script)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/isaac-0414/UniversalKG.git
   cd UniversalKG
   ```

2. Create a `.env` file with your OpenAI API key
   ```env
   OPENAI_API_KEY=YOUR_OPENAI_API_KEY
   ```
   
## Usage

### Basic Usage

1. Test UKG generation
   1) Open `main.py`, locate this line, and change this to the source text you want to test upon
     ```python
      # change this to the source text you want to test upon
      source_document_path = '/examples/source2'
     ```
   2) Run `python3 main.py` if your are using MacOS and `python main.py` if you are on a Windows machine.

2. Test querying existing UKG
   1) Open `qa.py`, locate these 2 lines, and change them to the question you want to ask and the path your KG is stored
     ```python
      question = "Who is the Chancellor of UIUC from 2015-2016?"
      kg_path = './kg_save/knowledge_graph.pkl'
     ```
   2) Run `python3 qa.py` if you are using MacOS and `python qa.py` if you are on a Windows machine.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
