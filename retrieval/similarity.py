import numpy as np


def compute_cosine_similarity(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    db_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
    return np.dot(db_norm, query_norm)


def compute_euclidean_similarity(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(database - query, axis=1)
    max_dist = np.max(distances)
    if max_dist > 0:
        return 1.0 - distances / max_dist
    return 1.0 - distances
