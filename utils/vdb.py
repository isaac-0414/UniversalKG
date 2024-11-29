"""
Implementation of a vector database, not used in the final version
"""
from utils.similarity import cosine_similarity
import json
import os

class VDB:
    """
    Class to manage a vector database, currently one vector database is stored in one JSON file
    """
    def __init__(self, vdb_file: str, empty_db=True):
        self.vdb_file = vdb_file
        if empty_db:
            self.empty_db()

    # query the vector of the id
    def query_id(self, id: str) -> list[float]:
        """
        query the vector of the id

        Parameters:
        id (str): id of interest

        Returns:
        list[float]: vector corresponding to id
        """
        with open(self.vdb_file, 'r') as infile:
            data = json.load(infile)
        return data[id]
        

    def query_index(self, input_vector: list[float], count: int=15) -> list[dict]:
        """
        query the most similar vectors from the vector database

        Parameters:
        input_vector (list[float]): the vector to compare to
        count (int): number of vectors want

        Returns:
         [{'id': str, 'score': float}]: a list of id's and their cosine similarity with input
        """
        with open(self.vdb_file, 'r') as infile:
            data = json.load(infile)
        
        scores = list()
        for id, vector in data.items():
            score = cosine_similarity(input_vector, vector)
            #print(score)
            scores.append({'id': id, 'score': score})
        ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
        return ordered[0:count]


    def insert_index(self, in_data: {str: list[float]}) -> None:
        """
        Insert data into the vector database

        Parameters:
        in_data ({str: list[float]}): Dictionary maps id to the vector
        """
        data = {}
        # Read existing data from file
        with open(self.vdb_file, 'r') as infile:
            data = json.load(infile)

        # append new data to the old data
        data.update(in_data)

        # Write updated data back to file
        with open(self.vdb_file, 'w') as outfile:
            json.dump(data, outfile, indent=2)

    def empty_db(self) -> None:
        """
        empty the current database file
        """
        with open(self.vdb_file, 'w') as f:
            json.dump({}, f)