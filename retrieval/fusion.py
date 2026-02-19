import numpy as np


def late_fusion(
    text_scores: np.ndarray,
    image_scores: np.ndarray,
    text_weight: float = 0.4,
    score_mode: str = "none"
) -> np.ndarray:
    text_scores = normalize_scores(text_scores, score_mode)
    image_scores = normalize_scores(image_scores, score_mode)
    
    fused = text_weight * text_scores + (1 - text_weight) * image_scores
    return fused


def early_fusion(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    text_weight: float = 1.0,
    image_weight: float = 1.0
) -> np.ndarray:
    text_vec = text_embeddings * text_weight
    image_vec = image_embeddings * image_weight
    fused = np.concatenate([text_vec, image_vec], axis=-1)
    
    norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8
    return fused / norms


def normalize_scores(scores: np.ndarray, mode: str = "none") -> np.ndarray:
    if mode == "none":
        return scores
    if mode == "zscore":
        mean = float(scores.mean())
        std = float(scores.std())
        return (scores - mean) / (std + 1e-8)
    if mode == "minmax":
        min_val = float(scores.min())
        max_val = float(scores.max())
        return (scores - min_val) / (max_val - min_val + 1e-8)
    return scores
