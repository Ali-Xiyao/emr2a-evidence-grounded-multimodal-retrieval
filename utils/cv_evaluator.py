import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils.metrics import (
    compute_confusion_matrix,
    compute_precision_recall_f1,
    compute_top_k_accuracy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CVRetrievalEvaluator:
    def __init__(
        self,
        cv_folds: int = 5,
        pca_dim: int = 128,
        top_k: int = 5,
        seed: int = 42,
    ):
        self.cv_folds = cv_folds
        self.pca_dim = pca_dim
        self.top_k = top_k
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.random = random.Random(seed)
    
    def stratified_split(
        self,
        patient_ids: List[str],
        labels: List[str],
    ) -> List[Tuple[List[str], List[str]]]:
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        splits = []
        
        for train_idx, test_idx in skf.split(patient_ids, labels):
            train_ids = [patient_ids[i] for i in train_idx]
            test_ids = [patient_ids[i] for i in test_idx]
            splits.append((train_ids, test_ids))
        
        return splits
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def process_embeddings(
        self,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_embeddings)
        test_scaled = scaler.transform(test_embeddings)
        
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
    
    def concat_fusion(
        self,
        img_vec: np.ndarray,
        txt_vec: np.ndarray,
    ) -> np.ndarray:
        fused = np.concatenate([img_vec, txt_vec], axis=1)
        return self._normalize_rows(fused)
    
    def compute_cosine_similarity(
        self,
        query_vec: np.ndarray,
        db_vecs: np.ndarray,
    ) -> np.ndarray:
        return np.dot(db_vecs, query_vec)
    
    def retrieve_topk(
        self,
        query_vec: np.ndarray,
        db_vecs: np.ndarray,
        db_labels: List[str],
        top_k: int,
        db_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[float], List[str]]:
        similarities = self.compute_cosine_similarity(query_vec, db_vecs)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_labels = [db_labels[i] for i in top_indices]
        top_scores = [float(similarities[i]) for i in top_indices]
        if db_ids:
            top_patient_ids = [db_ids[i] for i in top_indices]
        else:
            top_patient_ids = [f"neighbor_{i}" for i in top_indices]
        return top_labels, top_scores, top_patient_ids
    
    def compute_vote_accuracy(
        self,
        top_labels: List[List[str]],
        top_scores: List[List[float]],
        true_labels: List[str],
        weighted: bool = False,
    ) -> float:
        correct = 0
        for labels, scores, true_label in zip(top_labels, top_scores, true_labels):
            if weighted:
                label_to_score = {}
                for label, score in zip(labels, scores):
                    if label not in label_to_score:
                        label_to_score[label] = 0.0
                    label_to_score[label] += score
                predicted = max(label_to_score.items(), key=lambda x: x[1])[0]
            else:
                from collections import Counter
                predicted = Counter(labels).most_common(1)[0][0]
            
            if predicted == true_label:
                correct += 1
        
        return correct / len(true_labels)
    
    def evaluate_fold(
        self,
        train_img: Optional[np.ndarray],
        train_txt: Optional[np.ndarray],
        test_img: Optional[np.ndarray],
        test_txt: Optional[np.ndarray],
        train_labels: List[str],
        test_labels: List[str],
        test_ids: List[str],
        fusion: str = "concat",
        top_k_list: Optional[List[int]] = None,
        w_text: float = 0.5,
        train_ids: Optional[List[str]] = None,
    ) -> Dict:
        if top_k_list is None:
            top_k_list = [1, 3, 5, self.top_k]
        
        results = {}
        
        train_img_proc: Optional[np.ndarray] = None
        test_img_proc: Optional[np.ndarray] = None
        train_txt_proc: Optional[np.ndarray] = None
        test_txt_proc: Optional[np.ndarray] = None

        if train_img is not None and test_img is not None:
            train_img_proc, test_img_proc = self.process_embeddings(train_img, test_img)
        if train_txt is not None and test_txt is not None:
            train_txt_proc, test_txt_proc = self.process_embeddings(train_txt, test_txt)
        
        if fusion == "image_only":
            if train_img_proc is None or test_img_proc is None:
                raise ValueError("image_only fusion requires image embeddings")
            db_vecs = train_img_proc
            query_vecs = test_img_proc
        elif fusion == "text_only":
            if train_txt_proc is None or test_txt_proc is None:
                raise ValueError("text_only fusion requires text embeddings")
            db_vecs = train_txt_proc
            query_vecs = test_txt_proc
        elif fusion == "concat":
            if (
                train_img_proc is None
                or test_img_proc is None
                or train_txt_proc is None
                or test_txt_proc is None
            ):
                raise ValueError("concat fusion requires both image and text embeddings")
            db_vecs = self.concat_fusion(train_img_proc, train_txt_proc)
            query_vecs = self.concat_fusion(test_img_proc, test_txt_proc)
        elif fusion == "late":
            if (
                train_img_proc is None
                or test_img_proc is None
                or train_txt_proc is None
                or test_txt_proc is None
            ):
                raise ValueError("late fusion requires both image and text embeddings")
            db_vecs = None
            query_vecs = None
            train_img_vecs = train_img_proc
            train_txt_vecs = train_txt_proc
            test_img_vecs = test_img_proc
            test_txt_vecs = test_txt_proc
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")
        
        topk_results = {f"top{k}": [] for k in top_k_list}
        all_top_labels = []
        all_top_scores = []
        all_top_patient_ids = []
        all_pred_top1 = []
        all_pred_vote = []
        all_pred_weighted = []
        
        if fusion == "late":
            for i, (test_img_vec, test_txt_vec) in enumerate(zip(test_img_vecs, test_txt_vecs)):
                img_similarities = self.compute_cosine_similarity(test_img_vec, train_img_vecs)
                txt_similarities = self.compute_cosine_similarity(test_txt_vec, train_txt_vecs)
                combined_similarities = w_text * txt_similarities + (1 - w_text) * img_similarities
                
                top_indices = np.argsort(combined_similarities)[-self.top_k:][::-1]
                top_labels = [train_labels[idx] for idx in top_indices]
                top_scores = [float(combined_similarities[idx]) for idx in top_indices]
                if train_ids:
                    top_patient_ids = [train_ids[idx] for idx in top_indices]
                else:
                    top_patient_ids = [f"neighbor_{idx}" for idx in top_indices]
                
                all_top_labels.append(top_labels)
                all_top_scores.append(top_scores)
                all_top_patient_ids.append(top_patient_ids)
                
                all_pred_top1.append(top_labels[0])
                
                from collections import Counter
                vote_pred = Counter(top_labels).most_common(1)[0][0]
                all_pred_vote.append(vote_pred)
                
                label_to_score = {}
                for label, score in zip(top_labels, top_scores):
                    if label not in label_to_score:
                        label_to_score[label] = 0.0
                    label_to_score[label] += score
                weighted_pred = max(label_to_score.items(), key=lambda x: x[1])[0]
                all_pred_weighted.append(weighted_pred)
                
                for k in top_k_list:
                    if test_labels[i] in top_labels[:k]:
                        topk_results[f"top{k}"].append(1)
                    else:
                        topk_results[f"top{k}"].append(0)
        else:
            for i, query_vec in enumerate(query_vecs):
                if train_ids:
                    top_labels, top_scores, top_patient_ids = self.retrieve_topk(
                        query_vec, db_vecs, train_labels, self.top_k, train_ids
                    )
                else:
                    top_labels, top_scores, top_patient_ids = self.retrieve_topk(
                        query_vec, db_vecs, train_labels, self.top_k
                    )
                all_top_labels.append(top_labels)
                all_top_scores.append(top_scores)
                all_top_patient_ids.append(top_patient_ids)
                
                all_pred_top1.append(top_labels[0])
                
                from collections import Counter
                vote_pred = Counter(top_labels).most_common(1)[0][0]
                all_pred_vote.append(vote_pred)
                
                label_to_score = {}
                for label, score in zip(top_labels, top_scores):
                    if label not in label_to_score:
                        label_to_score[label] = 0.0
                    label_to_score[label] += score
                weighted_pred = max(label_to_score.items(), key=lambda x: x[1])[0]
                all_pred_weighted.append(weighted_pred)
                
                for k in top_k_list:
                    if test_labels[i] in top_labels[:k]:
                        topk_results[f"top{k}"].append(1)
                    else:
                        topk_results[f"top{k}"].append(0)
        
        for k in top_k_list:
            results[f"top{k}"] = np.mean(topk_results[f"top{k}"])
        
        results["vote_acc"] = self.compute_vote_accuracy(
            all_top_labels, all_top_scores, test_labels, weighted=False
        )
        results["weighted_vote_acc"] = self.compute_vote_accuracy(
            all_top_labels, all_top_scores, test_labels, weighted=True
        )
        
        labels = sorted(list(set(train_labels + test_labels)))
        
        precision_recall_f1 = compute_precision_recall_f1(
            all_pred_vote, test_labels, labels
        )
        
        results["macro_precision"] = np.mean([v["precision"] for v in precision_recall_f1.values()])
        results["macro_recall"] = np.mean([v["recall"] for v in precision_recall_f1.values()])
        results["macro_f1"] = np.mean([v["f1"] for v in precision_recall_f1.values()])
        
        results["confusion_matrix_top1"] = compute_confusion_matrix(
            all_pred_top1, test_labels, labels
        )
        results["confusion_matrix_vote"] = compute_confusion_matrix(
            all_pred_vote, test_labels, labels
        )
        
        results["all_top_labels"] = all_top_labels
        results["all_top_scores"] = all_top_scores
        results["all_top_patient_ids"] = all_top_patient_ids
        results["test_patient_ids"] = test_ids
        
        return results
    
    def run_cv(
        self,
        patient_ids: List[str],
        labels: List[str],
        embeddings: Dict[str, Dict[str, np.ndarray]],
        fusion: str = "concat",
        top_k_list: Optional[List[int]] = None,
        w_text: float = 0.5,
    ) -> Dict:
        splits = self.stratified_split(patient_ids, labels)
        pid_to_label = dict(zip(patient_ids, labels))
        
        all_results = []
        for fold_idx, (train_ids, test_ids) in enumerate(splits):
            logger.info(f"Processing fold {fold_idx + 1}/{self.cv_folds}")
            logger.info(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
            
            train_labels = [pid_to_label[pid] for pid in train_ids]
            test_labels = [pid_to_label[pid] for pid in test_ids]
            
            label_counts = {}
            for label in train_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            logger.info(f"Train label distribution: {label_counts}")
            
            train_img: Optional[np.ndarray] = None
            test_img: Optional[np.ndarray] = None
            train_txt: Optional[np.ndarray] = None
            test_txt: Optional[np.ndarray] = None

            if fusion in {"concat", "image_only", "late"}:
                train_img = np.stack([embeddings[pid]["image"] for pid in train_ids])
                test_img = np.stack([embeddings[pid]["image"] for pid in test_ids])
            if fusion in {"concat", "text_only", "late"}:
                train_txt = np.stack([embeddings[pid]["text"] for pid in train_ids])
                test_txt = np.stack([embeddings[pid]["text"] for pid in test_ids])
            
            fold_results = self.evaluate_fold(
                train_img, train_txt, test_img, test_txt,
                train_labels, test_labels, test_ids, fusion, top_k_list, w_text, train_ids
            )
            fold_results["fold"] = fold_idx + 1
            fold_results["train_ids"] = train_ids
            all_results.append(fold_results)
            
            logger.info(f"Fold {fold_idx + 1} results: Top1={fold_results['top1']:.4f}, "
                       f"Vote Acc={fold_results['vote_acc']:.4f}, "
                       f"Weighted Acc={fold_results['weighted_vote_acc']:.4f}")
        
        summary = self._compute_summary(all_results)
        return {
            "fold_results": all_results,
            "summary": summary,
        }
    
    def _compute_summary(self, all_results: List[Dict]) -> Dict:
        summary = {}
        metrics = ["top1", "top3", "top5", "vote_acc", "weighted_vote_acc", 
                   "macro_precision", "macro_recall", "macro_f1"]
        
        for metric in metrics:
            values = [r[metric] for r in all_results]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        
        return summary
    
    def save_results(
        self,
        results: Dict,
        output_dir: Path,
        experiment_id: str,
        config: Dict,
    ):
        output_dir = Path(output_dir)
        exp_dir = output_dir / f"exp_{experiment_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = exp_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        for fold_result in results["fold_results"]:
            fold_dir = exp_dir / f"fold_{fold_result['fold']}"
            fold_dir.mkdir(exist_ok=True)
            
            fold_result_serializable = self._make_serializable(fold_result)
            metrics_path = fold_dir / "metrics.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(fold_result_serializable, f, ensure_ascii=False, indent=2)
        
        summary_path = exp_dir / "summary.csv"
        self._save_summary_csv(results["summary"], summary_path)
        
        if "vlm_review" in results:
            vlm_review_path = exp_dir / "vlm_review_summary.json"
            with vlm_review_path.open("w", encoding="utf-8") as f:
                json.dump(results["vlm_review"], f, ensure_ascii=False, indent=2)
        
        self._plot_confusion_matrices(results, exp_dir)
        
        logger.info(f"Results saved to {exp_dir}")
    
    def _save_summary_csv(self, summary: Dict, output_path: Path):
        import csv
        
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Mean", "Std", "Min", "Max"])
            
            for metric, stats in summary.items():
                writer.writerow([
                    metric,
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}",
                ])
    
    def _plot_confusion_matrices(self, results: Dict, output_dir: Path):
        labels = sorted(list(set(
            k for r in results["fold_results"] 
            for k in r["confusion_matrix_top1"].keys()
        )))
        
        avg_cm_top1 = np.zeros((len(labels), len(labels)))
        avg_cm_vote = np.zeros((len(labels), len(labels)))
        
        for fold_result in results["fold_results"]:
            cm_top1 = np.array([
                [fold_result["confusion_matrix_top1"][true][pred] for pred in labels]
                for true in labels
            ])
            cm_vote = np.array([
                [fold_result["confusion_matrix_vote"][true][pred] for pred in labels]
                for true in labels
            ])
            avg_cm_top1 += cm_top1
            avg_cm_vote += cm_vote
        
        avg_cm_top1 /= len(results["fold_results"])
        avg_cm_vote /= len(results["fold_results"])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.heatmap(avg_cm_top1, annot=True, fmt=".1f", cmap="Blues", 
                   xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title("Confusion Matrix (Top1)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        
        sns.heatmap(avg_cm_vote, annot=True, fmt=".1f", cmap="Blues",
                   xticklabels=labels, yticklabels=labels, ax=axes[1])
        axes[1].set_title("Confusion Matrix (Vote)")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Confusion matrices saved to {output_dir / 'confusion_matrices.png'}")
