from typing import Dict, List

import numpy as np


def compute_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == gt)
    return correct / len(ground_truth)


def compute_top_k_accuracy(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int,
) -> float:
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    correct = 0
    for pred_list, gt in zip(predictions, ground_truth):
        if gt in pred_list[:k]:
            correct += 1
    
    return correct / len(ground_truth)


def compute_precision_recall_f1(
    predictions: List[str],
    ground_truth: List[str],
    labels: List[str],
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    
    for label in labels:
        tp = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == label and gt == label)
        fp = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == label and gt != label)
        fn = sum(1 for pred, gt in zip(predictions, ground_truth) if pred != label and gt == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for gt in ground_truth if gt == label),
        }
    
    return metrics


def compute_confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    labels: List[str],
) -> Dict[str, Dict[str, int]]:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    
    for pred, gt in zip(predictions, ground_truth):
        if pred in label_to_idx and gt in label_to_idx:
            matrix[label_to_idx[gt], label_to_idx[pred]] += 1
    
    confusion_dict = {}
    for i, true_label in enumerate(labels):
        confusion_dict[true_label] = {}
        for j, pred_label in enumerate(labels):
            confusion_dict[true_label][pred_label] = int(matrix[i, j])
    
    return confusion_dict
