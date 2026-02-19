import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def concat_embeddings(
    text_emb: np.ndarray,
    image_emb: np.ndarray,
    text_weight: float = 1.0,
    image_weight: float = 1.0,
) -> np.ndarray:
    text_weight = float(text_weight)
    image_weight = float(image_weight)
    text_vec = text_emb * text_weight
    image_vec = image_emb * image_weight
    fused = np.concatenate([text_vec, image_vec], axis=0)
    return l2_normalize(fused)
