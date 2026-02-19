"""
Audit Metrics Module for VLM Review

This module implements metrics for evaluating the VLM as an audit/quality control module:
1. Selective Prediction: Coverage vs Accuracy on accepted samples
2. Error Detection: AUROC for detecting main pipeline errors
3. Abstain Quality: Precision/Recall of abstain decisions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
import logging

logger = logging.getLogger(__name__)


class SelectivePredictionMetrics:
    """
    Selective prediction metrics: Coverage vs Accuracy trade-off.
    
    The audit module can choose to accept or reject predictions.
    - Coverage: proportion of samples accepted
    - Accuracy on accepted: accuracy of main pipeline on accepted samples
    """
    
    def __init__(self):
        self.coverage_curve: List[float] = []
        self.accuracy_curve: List[float] = []
        self.thresholds: List[float] = []
        self.aurc: float = 0.0  # Area Under Risk-Coverage curve
    
    def compute(
        self,
        main_predictions: List[str],
        ground_truth: List[str],
        audit_decisions: List[str],  # 'accept', 'reject', 'abstain'
        audit_confidences: List[float],
    ) -> Dict:
        """
        Compute selective prediction metrics at various thresholds.
        
        Args:
            main_predictions: Predictions from main pipeline (retrieval+vote)
            ground_truth: True labels
            audit_decisions: Audit module decisions
            audit_confidences: Confidence scores from audit module
        
        Returns:
            Dictionary with coverage curve, accuracy curve, and AURC
        """
        n_samples = len(main_predictions)
        
        # Convert decisions to acceptance scores
        # accept=1, reject/abstain=0
        acceptance_scores = []
        for decision, conf in zip(audit_decisions, audit_confidences):
            if decision == "accept":
                acceptance_scores.append(conf)
            else:
                acceptance_scores.append(0.0)
        
        # Sort by acceptance score (descending)
        sorted_indices = np.argsort(acceptance_scores)[::-1]
        
        self.coverage_curve = []
        self.accuracy_curve = []
        self.thresholds = []
        
        # Compute metrics at each possible coverage level
        for k in range(1, n_samples + 1):
            accepted_indices = sorted_indices[:k]
            coverage = k / n_samples
            
            # Accuracy on accepted samples
            accepted_correct = sum(
                1 for idx in accepted_indices
                if main_predictions[idx] == ground_truth[idx]
            )
            accuracy_on_accepted = accepted_correct / k if k > 0 else 0.0
            
            self.coverage_curve.append(coverage)
            self.accuracy_curve.append(accuracy_on_accepted)
            self.thresholds.append(acceptance_scores[sorted_indices[k-1]])
        
        # Compute AURC (Area Under Risk-Coverage curve)
        # Risk = 1 - Accuracy
        risks = [1.0 - acc for acc in self.accuracy_curve]
        self.aurc = np.trapz(risks, self.coverage_curve)
        
        return {
            "coverage_curve": self.coverage_curve,
            "accuracy_curve": self.accuracy_curve,
            "thresholds": self.thresholds,
            "aurc": self.aurc,
        }
    
    def get_metrics_at_coverage(self, target_coverage: float) -> Dict:
        """Get accuracy at a specific coverage level."""
        if not self.coverage_curve:
            return {"accuracy": 0.0, "threshold": 0.0}
        
        # Find closest coverage
        idx = np.argmin(np.abs(np.array(self.coverage_curve) - target_coverage))
        return {
            "coverage": self.coverage_curve[idx],
            "accuracy": self.accuracy_curve[idx],
            "threshold": self.thresholds[idx],
        }


class ErrorDetectionMetrics:
    """
    Error detection metrics: Treat audit module as a binary classifier
    for detecting when main pipeline makes errors.
    """
    
    def __init__(self):
        self.auroc: float = 0.0
        self.optimal_threshold: float = 0.5
        self.f1_at_optimal: float = 0.0
        self.precision_curve: List[float] = []
        self.recall_curve: List[float] = []
    
    def compute(
        self,
        main_predictions: List[str],
        ground_truth: List[str],
        audit_scores: List[float],  # Higher = more likely to be error
    ) -> Dict:
        """
        Compute error detection metrics.
        
        Args:
            main_predictions: Predictions from main pipeline
            ground_truth: True labels
            audit_scores: Scores indicating likelihood of error (higher = more suspicious)
        
        Returns:
            Dictionary with AUROC, optimal threshold, F1, etc.
        """
        # Binary labels: 1 if main pipeline made error, 0 if correct
        y_true = np.array([
            1.0 if pred != true else 0.0
            for pred, true in zip(main_predictions, ground_truth)
        ])
        y_scores = np.array(audit_scores)
        
        # AUROC
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present in error detection, AUROC undefined")
            self.auroc = 0.5
        else:
            self.auroc = roc_auc_score(y_true, y_scores)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        self.precision_curve = precision.tolist()
        self.recall_curve = recall.tolist()
        
        # Find optimal threshold by F1
        f1_scores = []
        thresholds_to_try = np.linspace(0, 1, 101)
        for thresh in thresholds_to_try:
            y_pred = (y_scores >= thresh).astype(int)
            if np.sum(y_pred) == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        best_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds_to_try[best_idx]
        self.f1_at_optimal = f1_scores[best_idx]
        
        return {
            "auroc": self.auroc,
            "optimal_threshold": self.optimal_threshold,
            "f1_at_optimal": self.f1_at_optimal,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_thresholds": roc_thresholds.tolist(),
        }


class AbstainQualityMetrics:
    """
    Metrics for evaluating the quality of abstain decisions.
    
    Good abstain should:
    - Reject more errors than correct predictions (high precision on errors)
    - Not reject too many correct predictions (low false rejection rate)
    """
    
    def __init__(self):
        self.error_rejection_rate: float = 0.0  # % of errors that were rejected
        self.correct_rejection_rate: float = 0.0  # % of correct that were wrongly rejected
        self.abstain_precision: float = 0.0  # Precision of abstain decisions
        self.abstain_recall: float = 0.0  # Recall of abstain for errors
    
    def compute(
        self,
        main_predictions: List[str],
        ground_truth: List[str],
        audit_decisions: List[str],  # 'accept', 'reject', 'abstain'
    ) -> Dict:
        """
        Compute abstain quality metrics.
        
        Args:
            main_predictions: Predictions from main pipeline
            ground_truth: True labels
            audit_decisions: Audit module decisions
        
        Returns:
            Dictionary with abstain quality metrics
        """
        n_samples = len(main_predictions)
        
        # Identify errors and correct predictions
        is_error = [pred != true for pred, true in zip(main_predictions, ground_truth)]
        is_correct = [pred == true for pred, true in zip(main_predictions, ground_truth)]
        
        # Identify rejected samples
        is_rejected = [decision in ["reject", "abstain"] for decision in audit_decisions]
        
        # Error rejection rate: % of errors that were caught
        n_errors = sum(is_error)
        errors_rejected = sum(1 for e, r in zip(is_error, is_rejected) if e and r)
        self.error_rejection_rate = errors_rejected / n_errors if n_errors > 0 else 0.0
        
        # Correct rejection rate: % of correct predictions wrongly rejected
        n_correct = sum(is_correct)
        correct_rejected = sum(1 for c, r in zip(is_correct, is_rejected) if c and r)
        self.correct_rejection_rate = correct_rejected / n_correct if n_correct > 0 else 0.0
        
        # Abstain precision: % of rejected samples that were actually errors
        n_rejected = sum(is_rejected)
        rejected_were_errors = sum(1 for e, r in zip(is_error, is_rejected) if e and r)
        self.abstain_precision = rejected_were_errors / n_rejected if n_rejected > 0 else 0.0
        
        # Abstain recall: % of errors that were rejected
        self.abstain_recall = self.error_rejection_rate
        
        # F1 for abstain
        if self.abstain_precision + self.abstain_recall > 0:
            abstain_f1 = 2 * self.abstain_precision * self.abstain_recall / (self.abstain_precision + self.abstain_recall)
        else:
            abstain_f1 = 0.0
        
        return {
            "error_rejection_rate": self.error_rejection_rate,
            "correct_rejection_rate": self.correct_rejection_rate,
            "abstain_precision": self.abstain_precision,
            "abstain_recall": self.abstain_recall,
            "abstain_f1": abstain_f1,
            "n_errors": n_errors,
            "n_correct": n_correct,
            "n_rejected": n_rejected,
            "errors_rejected": errors_rejected,
            "correct_rejected": correct_rejected,
        }


class AuditMetricsAggregator:
    """
    Aggregates all audit metrics for comprehensive evaluation.
    """
    
    def __init__(self):
        self.selective = SelectivePredictionMetrics()
        self.error_detection = ErrorDetectionMetrics()
        self.abstain_quality = AbstainQualityMetrics()
    
    def compute_all(
        self,
        main_predictions: List[str],
        ground_truth: List[str],
        audit_decisions: List[str],
        audit_confidences: List[float],
        audit_scores: List[float],  # Error likelihood scores
    ) -> Dict:
        """
        Compute all audit metrics.
        
        Returns:
            Dictionary with all metrics
        """
        selective_results = self.selective.compute(
            main_predictions, ground_truth, audit_decisions, audit_confidences
        )
        
        error_detection_results = self.error_detection.compute(
            main_predictions, ground_truth, audit_scores
        )
        
        abstain_results = self.abstain_quality.compute(
            main_predictions, ground_truth, audit_decisions
        )
        
        # Get metrics at specific coverage levels
        coverage_80 = self.selective.get_metrics_at_coverage(0.80)
        coverage_90 = self.selective.get_metrics_at_coverage(0.90)
        coverage_95 = self.selective.get_metrics_at_coverage(0.95)
        
        return {
            "selective_prediction": selective_results,
            "error_detection": error_detection_results,
            "abstain_quality": abstain_results,
            "coverage_80": coverage_80,
            "coverage_90": coverage_90,
            "coverage_95": coverage_95,
            "summary": {
                "aurc": selective_results["aurc"],
                "auroc_error_detection": error_detection_results["auroc"],
                "error_rejection_rate": abstain_results["error_rejection_rate"],
                "correct_rejection_rate": abstain_results["correct_rejection_rate"],
                "abstain_f1": abstain_results["abstain_f1"],
            }
        }


def compute_audit_metrics(
    main_predictions: List[str],
    ground_truth: List[str],
    audit_decisions: List[str],
    audit_confidences: List[float],
    audit_error_scores: List[float],
) -> Dict:
    """
    Convenience function to compute all audit metrics.
    
    Args:
        main_predictions: Main pipeline predictions
        ground_truth: True labels
        audit_decisions: 'accept', 'reject', 'abstain'
        audit_confidences: Confidence in decision
        audit_error_scores: Likelihood of error (higher = more suspicious)
    
    Returns:
        Dictionary with all metrics
    """
    aggregator = AuditMetricsAggregator()
    return aggregator.compute_all(
        main_predictions,
        ground_truth,
        audit_decisions,
        audit_confidences,
        audit_error_scores,
    )
