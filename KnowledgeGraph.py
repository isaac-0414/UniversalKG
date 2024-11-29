from __future__ import annotations

import uuid
import openai
from utils.vdb import VDB
from utils.similarity import cosine_similarity
from utils.gpt import gpt3_embedding, gpt_chat
from typing import Union, Optional, Dict
from collections import deque

import networkx as nx
from pyvis.network import Network
import pyvis.network as nt


class KGRelation:
   """
   Relation class of Knowledge Graph, holds head and tail entity as string
   """
    
   def __init__(
      self,
      name: str,
      head_entity: str,
      tail_entity: str,
      data_properties: Dict[str, str],
      description: str,
      source: str
   ):
      self.id = str(uuid.uuid4())
      self.name: str = name
      self.head_entity: str = head_entity
      self.tail_entity: str = tail_entity
      self.data_properties: Dict[str, str] = data_properties
      self.description: str = description
      self.source: str = source
   
   def __str__(self):
      return f"\nRelation: (\n  id: {self.id}\n  name: {self.name}\n  head: {self.head_entity}\n  tail: {str(self.tail_entity)}\n  data_properties: {str(self.data_properties)}\n  description: {self.description}\n  source: {self.source}\n)\n"


class KGEntity:
   """
   Entity class of Knowledge Graph, holds a list of relations where this entity is head
   """
    
   def __init__(
      self,
      name: str,
      data_properties: Dict[str, str],
      description: str,
      types: list[str],
      relations: list[KGRelation]
   ):
      self.id: str = str(uuid.uuid4())
      self.name: str = name
      self.data_properties: Dict[str, str] = data_properties
      self.description: str = description
      self.types: list[str] = types
      self.relations: list[KGRelation] = relations

   def __str__(self):
      return_str = f"\nEntity: (\n  id: {self.id}\n  name: {self.name}\n  data_properties: {self.data_properties}\n  description: {self.description}\n  types: {str(self.types)}\n  relations: ["
      for relation in self.relations:
         return_str += relation.name + ','
      return_str += "]\n)\n"
      return return_str
   

class KnowledgeGraph:
   """
   The Knowledge Graph class, unlike usual Graph which only stores node, it stores both entities and 
   relations (nodes and edges). It also controls 3 vector databases which stores vectorized entities, 
   relations, and types.
   """

   def __init__(
      self,
      entities: Dict[str, KGEntity],
      relations: set[KGRelation],
      types: set[str],
      vdb_path: str='./vdb'
   ):
      self.entities: Dict[str, KGEntity] = entities
      self.relations: set[KGRelation] = relations
      self.entities_vdb_map: Dict[str, KGEntity] = dict()
      self.relations_vdb_map: Dict[str, KGRelation] = dict()
      self.types: set[str] = types
      self.entity_vdb: VDB = VDB(f'{vdb_path}/entity_vdb.json')
      self.relation_vdb: VDB = VDB(f'{vdb_path}/relation_vdb.json')
      self.types_vdb: VDB = VDB(f'{vdb_path}/types_vdb.json')

   
   def add_entity(self, entity: KGEntity) -> None:
      """
      Add entity into knowledge graph and vector database

      Parameters:
      entity (KGEntity): The entity to add
      """
      if not entity.name in self.entities:
         self.entities.update({entity.name: entity})
         self.entities_vdb_map.update({entity.id: entity})

         vector = gpt3_embedding(content=entity.name)

         self.entity_vdb.insert_index({entity.id: vector})
      else:
         self.entities[entity.name].description += " " + entity.description
      for type in entity.types:
         self.types.add(type)


   def add_relation(self, relation: KGRelation) -> None:
      """
      Add relation into knowledge graph and vector database

      Parameters:
      relation (KGRelation): The relation to add
      """
      self.relations.add(relation)
      self.relations_vdb_map.update({relation.id: relation})
      self.entities[relation.head_entity].relations.append(relation)

      vector = gpt3_embedding(content=relation.name)

      self.relation_vdb.insert_index({relation.id: vector})


   def __str__(self):
      return_str = "Knowledge Graph:\n\nEntities:\n"
      for entity_name, entity in self.entities.items():
         return_str += str(entity)
      return_str += "\nRelations:\n"
      for relation in self.relations:
         return_str += str(relation)
      return_str += "\nTypes:\n["
      for type in self.types:
         return_str += type + ","
      return_str += ']\n'
      return return_str
   


   def find_entity(self, entity_name: str) -> Optional[KGEntity]:
      """
      given name of an entity, find matching entity in the knowledge graph

      Parameters:
      entity_name (str): Name of the entity we want to find in KG

      Returns:
      Optional[KGEntity]: the result entity
      """
      entity = None

      if entity_name in self.entities:
         # If the name matches in the graph
         entity = self.entities[entity_name]
      else:
         # Find the most similar entity in the graph in terms of cosine similarity
         # If the similarity of the most similar entity is lower than 0.9, means no matching entity
         target_entity_vector = gpt3_embedding(content=entity_name)
         most_similar_pair = self.entity_vdb.query_index(input_vector=target_entity_vector, count=1)
         if most_similar_pair["score"] >= 0.90:
            entity = self.entities_vdb_map[most_similar_pair["id"]]
   
      if entity == None:
         return None    
      
      return entity
   


   def find_relation(self, head_name: str, tail_name: str, target_relation_name: str) -> Optional[KGRelation]:
      """
      Given a triplet of head, relation and tail, return the corresponding relation in knowledge graph

      Parameters:
      head_name (str): Name of the head entity
      tail_name (str): Name of the tail entity
      target_relation_name (str): name of the relation

      Returns:
      Optional[KGRelation]: the result relation
      """
      head = None
      # First find head entity in the knowledge graph
      if head_name in self.entities:
         # If the name matches in the graph
         head = self.entities[head_name]
      else:
         # Find the most similar entity in the graph in terms of cosine similarity
         # If the similarity of the most similar entity is lower than 0.9, means no matching entity
         target_head_entity_vector = gpt3_embedding(content=head_name)
         most_similar_pair = self.entity_vdb.query_index(input_vector=target_head_entity_vector, count=1)
         if most_similar_pair["score"] >= 0.90:
            head = self.entities_vdb_map[most_similar_pair["id"]]
   
      if head == None:
         return None    
      
      # Then traverse all the relations of head, and find the one with the same tail entity and relation name
      for relation in head.relations:
         if tail_name == relation.tail_entity:
            # If name of the tail matches
            if target_relation_name == relation.name:
               # If relation name also matches
               return relation
            else:
               # If relation name doesn't match, use cosine similarity
               target_relation_vector = gpt3_embedding(content=target_relation_name)
               curr_relation_vector = self.relation_vdb.query_id(relation.id)
               similarity = cosine_similarity(target_relation_vector, curr_relation_vector)
               if similarity < 0.90:
                  continue
               return relation
         else:
            # if name does not match, do cosine similarity again
            target_tail_entity_vector = gpt3_embedding(content=tail_name)
            curr_tail_entity_vector = self.entity_vdb.query_id(self.entities[relation.tail_entity].id)
            tail_entity_similarity = cosine_similarity(target_tail_entity_vector, curr_tail_entity_vector)
            if tail_entity_similarity >= 0.90:
               if target_relation_name == relation.name:
                  # If relation name matches
                  return relation
               else:
                  # If relation name doesn't match, use cosine similarity
                  target_relation_vector = gpt3_embedding(content=target_relation_name)
                  curr_relation_vector = self.relation_vdb.query_id(relation.id)
                  similarity = cosine_similarity(target_relation_vector, curr_relation_vector)
                  if similarity < 0.90:
                     continue
                  return relation
      
      return None
   
   def relation_completion(self) -> None:
      """
      For all the relations from a head entity to tail entity, there should be an inverse relation from
      the tail entity to the head entity, but since our graph is generated by LLM, sometimes there is only 
      relation in one direction. This function is to "complete" the knowledge graph by creating an inverse
      relation for all the relations in KG if there wasn't one.
      """
      visited = set()
      def bfs(start: KGEntity, visited: set) -> None:
         """
         Standard BFS algorithm to traverse the graph

         Parameters:
         start (KGEntity): the start point in the graph
         visited (set): the set of all the visited entities
         """
         q = deque()
         q.append(start)

         visited.add(start)
         while len(q) != 0:
            current = q.popleft()
            for relation in current.relations:
               next_entity = self.entities[relation.tail_entity]
               if not next_entity in visited:
                  has_inverse_relation = False
                  for next_relation in next_entity.relations:
                     if next_relation.tail_entity == relation.head_entity:
                        has_inverse_relation = True
                  if not has_inverse_relation:
                     # If doesn't have inverse relation, use GPT to generate one
                     messages = [{"role": "system", "content": "You are an expert in linguistics and knowledge graph. You will be given a relation between two entities, and you will output a name for the inverse relation between them. Output only the relation name"}, {"role": "user", "content": f"Head Entity:{relation.head_entity}\nTail Entity: {relation.tail_entity}\nRelation: {relation.name}"}]
                     inverse_relation = gpt_chat(messages, model="gpt-4")

                     # deal with formatting issues of GPT
                     if inverse_relation.startswith('Inverse Relation: '):
                        inverse_relation = inverse_relation[18:]
                     if not inverse_relation.endswith('Relation'):
                        inverse_relation += "_Relation"
                     
                     # add the new relation to the graph
                     self.add_relation(KGRelation(name=inverse_relation, head_entity=relation.tail_entity, tail_entity=relation.head_entity, data_properties=relation.data_properties, description=relation.description, source=relation.source))

                  q.append(next_entity)
                  visited.add(next_entity)
   
      return bfs(list(self.entities.values())[0], visited)
   
   
   def find_path(self, e1: KGEntity, e2: KGEntity) -> list[KGRelation]:
      """
      Given two entities in the graph, find a path from e1 to e2

      Parameters:
      e1 (KGEntity): the first entity
      e2 (KGEntity): the second entity

      Returns:
      list[KGRelation]: path as the list of relations
      """
      visited = set()
      def dfs_find_path(current: KGEntity, target: KGEntity, visited: set) -> list[KGRelation]:
         """
         DFS helper function

         Parameters:
         current (KGEntity): the current entity of this recursion
         target (KGEntity): the target entity
         visited (set): the set of all the visited entities

         Returns:
         list[KGRelation]: path as the list of relations
         """
         if current.id in visited: 
            # Base Case 1: visited entity
            return None
         if current.id == target.id: 
            # Base Case 2: find target
            return []

         visited.add(current.id)

         for relation in current.relations:
            next_entity = self.entities[relation.tail_entity]
            path_result = dfs_find_path(next_entity, target, visited)
            if path_result is not None:
               return [relation] + path_result
         
         return None
      return dfs_find_path(e1, e2, visited)
   
   
   def find_matching_entities(self, path: list[KGRelation], subgraph: KnowledgeGraph, question: str,) -> set[KGEntity]:
      """
      Given a path from one entity to the target entity [ENTITY] in the subgraph, follow the same path in the 
      knowledge graph and find the target entity in the knowledge graph.
      For all the relations that are not the last one, I used the names of head, relation, and tail to validate 
      them, for the last one, which contains target entity [ENTITY] as tail, I used GPT to validate it.

      Parameters:
      path (list[KGRelation]): path from an entity in subgraph to the unknown entity [ENTITY]
      subgraph (KnowledgeGraph): the subgraph
      question (str): The question currently dealing with, used to validate relation during recursion

      Returns:
      set[KGEntity]: the set of all possible target entities
      """
      visited = set()
      matching_entities = set() # will have the result of dfs_find_matching_entities

      def dfs_find_matching_entities(current: KGEntity, path_idx: int, visited: set) -> None:
         """
         Using DFS to find all the matching entities

         Parameters:
         current (KGEntity): the current entity for this recursion
         path_idx (int): the current index of path
         visited (set): set of all the visited entities
         """
         if current.id in visited: 
            return
         
         if path_idx == len(path):
            matching_entities.add(current)
            return

         visited.add(current.id)

         for relation in current.relations:
            path_relation_vector = subgraph.relation_vdb.query_id(path[path_idx].id)
            curr_relation_vector = self.relation_vdb.query_id(relation.id)
            similarity = cosine_similarity(path_relation_vector, curr_relation_vector)
            if similarity < 0.90:
               continue
            
            # if cosine similarity >= 0.9, means the relation in path and current relation in this recursion matches
      
            if path_idx != len(path) - 1:
               # If path not ended, keeping recursing

               # another step of validation, matching tail entity
               if path[path_idx].tail_entity == relation.tail_entity:
                  # if tail entities' names are also the same
                  next_entity = self.entities[relation.tail_entity]
                  dfs_find_matching_entities(next_entity, path_idx + 1, visited)
               else:
                  # name not same, cosine similarity again
                  input_tail_entity_vector = subgraph.entity_vdb.query_id(self.entities[path[path_idx].tail_entity].id)
                  tail_entity_vector = self.entity_vdb.query_id(self.entities[relation.tail_entity].id)
                  next_entity_similarity = cosine_similarity(input_tail_entity_vector, tail_entity_vector)
                  if next_entity_similarity >= 0.90:
                     next_entity = self.entities[relation.tail_entity]
                     dfs_find_matching_entities(next_entity, path_idx + 1, visited)
            else:
               # using GPT to validate the last relation
               system_prompt = "You are an expert in linguistics and knowledge graph. You will help determine that whether a relation is involved in the question. You should only output True or False."
               prompt = f"Question: {question}\n\n{str(relation)}"
               messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
               response = gpt_chat(messages=messages, model="gpt-4")
               if response == "True":
                  next_entity = self.entities[relation.tail_entity]
                  # this will reach the base case
                  dfs_find_matching_entities(next_entity, path_idx + 1, visited)
         
         return None
      
      # Find the starting entity in the graph
      if path[0].head_entity in self.entities:
         # If the name matches
         start_entity: KGEntity = self.entities[path[0].head_entity]
      else:
         # If it doesn't match, use cosine similarity again
         vector = subgraph.entity_vdb.query_id(self.entities[path[0].head_entity].id)
         most_similar = self.entity_vdb.query_index(vector, 1)[0]
        
         if most_similar['score'] < 0.90:
            # TODO: raise exception here just for testing
            raise Exception("Fail since one entity in question doesn't exist in KG")
            
         start_entity: KGEntity = self.entities_vdb_map[most_similar['id']]
   
      dfs_find_matching_entities(start_entity, 0, visited)
      return matching_entities
   
   
   def visualize(self, path="./kg_visualization/knowledge_graph.html") -> None:
      """
      Visualize the knowledge graph into HTML format

      Parameters:
      path (str): path of where you want to save the visualization
      """
      # Create a directed graph
      G = nx.DiGraph()

      # Add nodes and edges from both lists
      for relation in self.relations:
         G.add_edge(relation.head_entity, relation.tail_entity, label=relation.name)

      # Create a Network instance
      nt_graph = nt.Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="white")

      # Add nodes to the NetworkX graph
      for node in G.nodes():
         label = str(node)
         color = "blue" if label.startswith("l(") else "black"
         shape = "box" if color == "blue" else "ellipse"
         nt_graph.add_node(label, label=label, color=color, shape=shape)

      # Use a different layout algorithm with adjustable parameters
      pos = nx.spring_layout(G, k=0.1)  # Adjust the 'k' parameter to control the force

      # Add edges to the NetworkX graph
      for u, v in G.edges():
         label = G[u][v]['label']
         nt_graph.add_edge(str(u), str(v), title=label)

      # Show the graph in an HTML file
      nt_graph.write_html(path)
         
        
