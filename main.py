import pickle
import time
from utils.kg_gen import *
from KnowledgeGraph import *

# change this to the source text you want to test upon
source_document_path = '/examples/source2'

if __name__ == '__main__':
    start_time = time.time()
    with open (source_document_path , 'r') as f:
        text = f.read()
    text_chunks = split_text(text, 6000, 1500, '.')

    knowledge_graph = KnowledgeGraph(entities=dict(), relations=set(), types=set(), vdb_path='./vdb')

    iter = 0
    for text_chunk in text_chunks:
        print('iteration: ', iter)
        entities = entity_extract(text_chunk)
        for entity in entities:
            knowledge_graph.add_entity(entity)
        
        text_chunk = entity_disambiguation(text_chunk)
        
        relations = predicate_extract(text=text_chunk, entities=entities)
        for relation in relations:
            knowledge_graph.add_relation(relation)

        iter += 1
    
    knowledge_graph.relation_completion()
    print(knowledge_graph)
    with open('./kg_save/knowledge_graph.pkl', 'wb') as file:
        pickle.dump(knowledge_graph, file)

    knowledge_graph.visualize()
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")