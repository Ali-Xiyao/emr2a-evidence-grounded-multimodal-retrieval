#!/usr/bin/env python3
"""
Compute per-class metrics for CNN baseline results.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def compute_per_class_metrics(confusion_matrix_list, num_classes=4):
    """Compute per-class metrics from confusion matrices."""
    
    classes = ['Bacterial', 'Normal', 'PJP', 'Viral']
    
    all_metrics = {cls: {'precision': [], 'recall': [], 'f1': [], 'specificity': []} for cls in classes}
    
    for cm in confusion_matrix_list:
        for i, cls in enumerate(classes):
            tp = cm[i, i]
            # sklearn confusion_matrix convention: rows=true, cols=pred
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            all_metrics[cls]['precision'].append(precision)
            all_metrics[cls]['recall'].append(recall)
            all_metrics[cls]['f1'].append(f1)
            all_metrics[cls]['specificity'].append(specificity)
    
    return all_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute per-class metrics for CNN baseline")
    parser.add_argument("--results_path", default="outputs/cnn_baselines/exp_image_only/results.json",
                        help="Path to results.json")
    args = parser.parse_args()
    
    with open(args.results_path) as f:
        results = json.load(f)
    
    fold_results = results.get("fold_results", [])
    confusion_matrices = [np.array(fr["confusion_matrix"]) for fr in fold_results]
    
    all_metrics = compute_per_class_metrics(confusion_matrices)
    
    print("=" * 80)
    print("Per-Class Metrics (mean±std over 5 folds)")
    print("=" * 80)
    print()
    
    for cls in ['Bacterial', 'Normal', 'PJP', 'Viral']:
        print(f"### {cls}")
        print(f"  Precision:  {np.mean(all_metrics[cls]['precision']):.4f} ± {np.std(all_metrics[cls]['precision']):.4f}")
        print(f"  Recall:     {np.mean(all_metrics[cls]['recall']):.4f} ± {np.std(all_metrics[cls]['recall']):.4f}")
        print(f"  F1:         {np.mean(all_metrics[cls]['f1']):.4f} ± {np.std(all_metrics[cls]['f1']):.4f}")
        print(f"  Specificity: {np.mean(all_metrics[cls]['specificity']):.4f} ± {np.std(all_metrics[cls]['specificity']):.4f}")
        print()
    
    print("=" * 80)
    print("Markdown Table Format")
    print("=" * 80)
    print()
    print("| Class     | Precision (mean±std) | Recall (mean±std) | F1 (mean±std) | Specificity (mean±std) |")
    print("|-----------|---------------------|------------------|---------------|----------------------|")
    
    for cls in ['Bacterial', 'Normal', 'PJP', 'Viral']:
        prec = f"{np.mean(all_metrics[cls]['precision']):.4f}±{np.std(all_metrics[cls]['precision']):.4f}"
        rec = f"{np.mean(all_metrics[cls]['recall']):.4f}±{np.std(all_metrics[cls]['recall']):.4f}"
        f1 = f"{np.mean(all_metrics[cls]['f1']):.4f}±{np.std(all_metrics[cls]['f1']):.4f}"
        spec = f"{np.mean(all_metrics[cls]['specificity']):.4f}±{np.std(all_metrics[cls]['specificity']):.4f}"
        
        print(f"| {cls:9s} | {prec:19s} | {rec:16s} | {f1:13s} | {spec:20s} |")


if __name__ == "__main__":
    main()
