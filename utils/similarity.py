import numpy as np
from numpy.linalg import norm

def dot_product_similarity(v1: list[float], v2: list[float]) -> float: 
    """
    dot product of two vectors

    Parameters:
    v1 (list[float]): the first vector
    v2 (list[float]): the second vector

    Returns:
    float: dot product similarity
    """
    return np.dot(v1, v2)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    cosine similarity of two vectors
    
    Parameters:
    v1 (list[float]): the first vector
    v2 (list[float]): the second vector

    Returns:
    float: cosine similarity
    """
    return np.dot(v1, v2) / (norm(v1) * norm(v2))