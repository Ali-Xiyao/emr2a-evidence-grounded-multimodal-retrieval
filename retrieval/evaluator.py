import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .similarity import compute_cosine_similarity
from .fusion import late_fusion, early_fusion, normalize_scores


class RetrievalEvaluator:
    def __init__(
        self,
        test_ratio: float = 0.2,
        seed: int = 42,
        use_pca: bool = False,
        pca_dim: int = 128,
    ):
        self.test_ratio = test_ratio
        self.seed = seed
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.rng = random.Random(seed)
    
    def stratified_split(
        self,
        labels: List[str],
    ) -> Tuple[List[int], List[int]]:
        label_to_indices: Dict[str, List[int]] = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(str(label), []).append(idx)
        
        train_idx: List[int] = []
        test_idx: List[int] = []
        
        for _, idxs in label_to_indices.items():
            self.rng.shuffle(idxs)
            if len(idxs) <= 1:
                train_idx.extend(idxs)
                continue
            
            test_count = int(round(len(idxs) * self.test_ratio))
            test_count = max(1, min(test_count, len(idxs) - 1))
            test_idx.extend(idxs[:test_count])
            train_idx.extend(idxs[test_count:])
        
        return train_idx, test_idx
    
    def process_embeddings(
        self,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_embeddings)
        test_scaled = scaler.transform(test_embeddings)
        
        if not self.use_pca:
            return self._normalize_rows(train_scaled), self._normalize_rows(test_scaled)
        
        n_samples = train_scaled.shape[0]
        n_features = train_scaled.shape[1]
        n_components = min(self.pca_dim, n_samples - 1, n_features)
        
        if n_components <= 0:
            return self._normalize_rows(train_scaled), self._normalize_rows(test_scaled)
        
        pca = PCA(n_components=n_components)
        train_reduced = pca.fit_transform(train_scaled)
        test_reduced = pca.transform(test_scaled)
        
        return self._normalize_rows(train_reduced), self._normalize_rows(test_reduced)
    
    def _normalize_rows(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        return arr / norms
    
    def align_dims(
        self,
        train_text: Optional[np.ndarray],
        test_text: Optional[np.ndarray],
        train_image: Optional[np.ndarray],
        test_image: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if train_text is not None and test_text is not None:
            train_text, test_text = self.process_embeddings(train_text, test_text)
        
        if train_image is not None and test_image is not None:
            train_image, test_image = self.process_embeddings(train_image, test_image)
        
        return train_text, test_text, train_image, test_image
    
    def evaluate_retrieval(
        self,
        train_text: Optional[np.ndarray],
        test_text: Optional[np.ndarray],
        train_image: Optional[np.ndarray],
        test_image: Optional[np.ndarray],
        train_labels: List[str],
        test_labels: List[str],
        text_weight: float = 0.4,
        fusion_type: str = "late",
        score_mode: str = "none",
        top_k_list: List[int] = [1, 3, 5],
    ) -> Dict:
        results = {}
        
        if fusion_type == "early":
            if train_text is None or test_text is None or train_image is None or test_image is None:
                raise ValueError("Early fusion requires both text and image embeddings")
            
            train_fused = early_fusion(train_text, train_image, text_weight, 1 - text_weight)
            test_fused = early_fusion(test_text, test_image, text_weight, 1 - text_weight)
            
            for top_k in top_k_list:
                acc = self._compute_top_k_accuracy(
                    train_fused, test_fused, train_labels, test_labels, top_k
                )
                results[f"top{top_k}"] = acc
            
            results["weighted"] = self._compute_weighted_accuracy(
                train_fused, test_fused, train_labels, test_labels
            )
        
        else:
            if test_text is not None and train_text is not None:
                for top_k in top_k_list:
                    acc = self._compute_top_k_accuracy(
                        train_text, test_text, train_labels, test_labels, top_k
                    )
                    results[f"text_top{top_k}"] = acc
                
                results["text_weighted"] = self._compute_weighted_accuracy(
                    train_text, test_text, train_labels, test_labels
                )
            
            if test_image is not None and train_image is not None:
                for top_k in top_k_list:
                    acc = self._compute_top_k_accuracy(
                        train_image, test_image, train_labels, test_labels, top_k
                    )
                    results[f"image_top{top_k}"] = acc
                
                results["image_weighted"] = self._compute_weighted_accuracy(
                    train_image, test_image, train_labels, test_labels
                )
            
            if test_text is not None and test_image is not None:
                fused_scores = []
                for i in range(len(test_labels)):
                    text_sim = compute_cosine_similarity(test_text[i], train_text)
                    image_sim = compute_cosine_similarity(test_image[i], train_image)
                    fused = late_fusion(text_sim, image_sim, text_weight, score_mode)
                    fused_scores.append(fused)
                
                fused_scores = np.array(fused_scores)
                
                for top_k in top_k_list:
                    acc = self._compute_top_k_accuracy_from_scores(
                        fused_scores, train_labels, test_labels, top_k
                    )
                    results[f"top{top_k}"] = acc
                
                results["weighted"] = self._compute_weighted_accuracy_from_scores(
                    fused_scores, train_labels, test_labels
                )
                
                # Save top-k labels for each sample (for C3a analysis)
                for top_k in [5]:  # Only save for TopK=5
                    all_top_labels = self.get_all_top_labels(
                        fused_scores, train_labels, test_labels, top_k
                    )
                    results[f"all_top_labels_top{top_k}"] = all_top_labels
        
        return results
    
    def _compute_top_k_accuracy(
        self,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: List[str],
        test_labels: List[str],
        top_k: int,
    ) -> float:
        correct = 0
        for i, test_emb in enumerate(test_embeddings):
            similarities = compute_cosine_similarity(test_emb, train_embeddings)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_labels = [train_labels[idx] for idx in top_indices]
            if test_labels[i] in top_labels:
                correct += 1
        return correct / len(test_labels)
    
    def _compute_top_k_accuracy_from_scores(
        self,
        scores: np.ndarray,
        train_labels: List[str],
        test_labels: List[str],
        top_k: int,
    ) -> float:
        correct = 0
        for i, test_scores in enumerate(scores):
            top_indices = np.argsort(test_scores)[-top_k:][::-1]
            top_labels = [train_labels[idx] for idx in top_indices]
            if test_labels[i] in top_labels:
                correct += 1
        return correct / len(test_labels)
    
    def _compute_weighted_accuracy(
        self,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: List[str],
        test_labels: List[str],
    ) -> float:
        correct = 0
        for i, test_emb in enumerate(test_embeddings):
            similarities = compute_cosine_similarity(test_emb, train_embeddings)
            top_indices = np.argsort(similarities)[-5:][::-1]
            top_labels = [train_labels[idx] for idx in top_indices]
            top_scores = [similarities[idx] for idx in top_indices]
            
            label_to_scores: Dict[str, float] = {}
            for label, score in zip(top_labels, top_scores):
                if label not in label_to_scores:
                    label_to_scores[label] = 0.0
                label_to_scores[label] += score
            
            predicted = max(label_to_scores.items(), key=lambda x: x[1])[0]
            if predicted == test_labels[i]:
                correct += 1
        return correct / len(test_labels)
    
    def _compute_weighted_accuracy_from_scores(
        self,
        scores: np.ndarray,
        train_labels: List[str],
        test_labels: List[str],
    ) -> float:
        correct = 0
        for i, test_scores in enumerate(scores):
            top_indices = np.argsort(test_scores)[-5:][::-1]
            top_labels = [train_labels[idx] for idx in top_indices]
            top_scores = [test_scores[idx] for idx in top_indices]
            
            label_to_scores: Dict[str, float] = {}
            for label, score in zip(top_labels, top_scores):
                if label not in label_to_scores:
                    label_to_scores[label] = 0.0
                label_to_scores[label] += score
            
            predicted = max(label_to_scores.items(), key=lambda x: x[1])[0]
            if predicted == test_labels[i]:
                correct += 1
        return correct / len(test_labels)

    def get_all_top_labels(
        self,
        scores: np.ndarray,
        train_labels: List[str],
        test_labels: List[str],
        top_k: int = 5,
    ) -> List[List[str]]:
        """Get top-k labels for each test sample.
        
        Returns:
            List of top-k labels for each test sample
        """
        all_top_labels = []
        for i, test_scores in enumerate(scores):
            top_indices = np.argsort(test_scores)[-top_k:][::-1]
            top_labels = [train_labels[idx] for idx in top_indices]
            all_top_labels.append(top_labels)
        return all_top_labels
