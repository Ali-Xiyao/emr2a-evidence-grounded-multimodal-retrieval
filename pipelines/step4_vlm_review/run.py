"""
Run VLM Audit Pipeline

This script runs the VLM-based audit/quality control for retrieval results.
The audit module evaluates whether retrieval+vote predictions are supported by evidence,
without making independent diagnoses.

Key Features:
- Audit decision: accept / reject / abstain
- Evidence citation: references specific TopK neighbors
- Selective prediction: Coverage vs Accuracy trade-off
- Error detection: AUROC for detecting main pipeline errors
- Abstain quality: Precision/Recall of abstain decisions
"""

import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from .vlm_audit_module import VLMAuditModule
from .audit_metrics import compute_audit_metrics
from data.manifest import load_manifest


def load_retrieval_results_from_folds(exp_dir: Path, max_samples: int = 30) -> Dict:
    """
    Load retrieval results from experiment folds.
    
    Args:
        exp_dir: Path to experiment directory
        max_samples: Maximum number of samples to process per fold
    
    Returns:
        Dict containing retrieval results
    """
    all_top_labels = []
    all_top_scores = []
    all_top_patient_ids = []
    test_patient_ids = []
    missing_detail_folds = []
    required_keys = ("all_top_labels", "all_top_scores", "test_patient_ids")
    
    for fold_idx in range(1, 6):
        fold_path = exp_dir / f"fold_{fold_idx}" / "metrics.json"
        if fold_path.exists():
            with open(fold_path, 'r', encoding='utf-8') as f:
                fold_data = json.load(f)

            missing_keys = [key for key in required_keys if key not in fold_data]
            if missing_keys:
                missing_detail_folds.append({
                    "path": str(fold_path),
                    "missing_keys": missing_keys,
                })
                logger.warning(f"Skipping {fold_path}: missing keys {missing_keys}")
                continue
            
            fold_top_labels = fold_data.get('all_top_labels', [])
            fold_top_scores = fold_data.get('all_top_scores', [])
            fold_top_patient_ids = fold_data.get('all_top_patient_ids', [])
            fold_test_ids = fold_data.get('test_patient_ids', [])
            
            sample_limit = min(
                len(fold_top_labels),
                len(fold_top_scores),
                len(fold_test_ids),
                max_samples,
            )
            for i in range(sample_limit):
                all_top_labels.append(fold_top_labels[i])
                all_top_scores.append(fold_top_scores[i])
                if i < len(fold_top_patient_ids):
                    all_top_patient_ids.append(fold_top_patient_ids[i])
                test_patient_ids.append(fold_test_ids[i])
    
    return {
        'all_top_labels': all_top_labels,
        'all_top_scores': all_top_scores,
        'all_top_patient_ids': all_top_patient_ids,
        'test_patient_ids': test_patient_ids,
        'missing_detail_folds': missing_detail_folds,
    }


def run_vlm_audit_pipeline(
    exp_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    max_samples: int = 30,
    model_path: str = None,
    device: str = "cuda",
    seed: int = 42,
    accept_threshold: float = 0.7,
    reject_threshold: float = 0.7,
    abstain_threshold: float = 0.5,
    conservative: bool = False,
):
    """
    Run VLM audit pipeline.
    
    Args:
        exp_dir: Path to experiment directory containing retrieval results
        manifest_path: Path to manifest file
        output_dir: Path to output directory
        max_samples: Maximum number of samples to process
        model_path: Path to VLM model
        device: Device to run VLM on
        seed: Random seed for sampling
        accept_threshold: Confidence threshold for accepting prediction
        reject_threshold: Confidence threshold for rejecting prediction
        abstain_threshold: Confidence threshold for abstaining
        conservative: Use conservative prompt (more abstain)
    """
    logger.info(f"Loading retrieval results from {exp_dir}")
    retrieval_data = load_retrieval_results_from_folds(exp_dir, max_samples)

    if not retrieval_data.get("test_patient_ids"):
        missing_detail_folds = retrieval_data.get("missing_detail_folds", [])
        hint = ""
        alt_dir = Path(f"{exp_dir}_v2")
        if alt_dir.exists():
            hint = f" Try --exp_dir {alt_dir}."
        if missing_detail_folds:
            example = missing_detail_folds[0]
            missing_keys = example.get("missing_keys", [])
            raise ValueError(
                "No per-sample retrieval results found. "
                f"{example.get('path')} missing keys {missing_keys}.{hint}"
            )
        raise ValueError(
            "No retrieval samples loaded from metrics.json. "
            "Check --exp_dir or re-run CV experiments to generate detailed metrics."
        )
    
    logger.info(f"Loading manifest from {manifest_path}")
    manifest = load_manifest(manifest_path)
    patient_id_to_record = {r.get("patient_id"): r for r in manifest}
    
    # Prepare data structures
    retrieval_predictions = []
    retrieval_similarities = []
    retrieval_labels = []
    top_neighbors_list = []
    query_patients = []
    sample_patient_ids = []
    
    all_top_labels = retrieval_data['all_top_labels']
    all_top_scores = retrieval_data['all_top_scores']
    all_top_patient_ids = retrieval_data.get('all_top_patient_ids', [])
    test_patient_ids = retrieval_data['test_patient_ids']

    total_samples = min(len(test_patient_ids), len(all_top_labels), len(all_top_scores))
    indices = list(range(total_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[:max_samples]
    logger.info(f"Sampling {len(indices)} cases from {total_samples} (seed={seed})")
    
    for idx in indices:
        test_pid = test_patient_ids[idx]
        if idx < len(all_top_labels):
            top_labels = all_top_labels[idx]
            top_scores = all_top_scores[idx]
            
            # Get majority vote prediction
            from collections import Counter
            vote_pred = Counter(top_labels).most_common(1)[0][0]
            retrieval_predictions.append(vote_pred)
            
            retrieval_similarities.append(top_scores)
            retrieval_labels.append(top_labels)
            
            # Build neighbor list
            neighbors = []
            if idx < len(all_top_patient_ids) and all_top_patient_ids[idx]:
                top_patient_ids = all_top_patient_ids[idx]
                for j, (label, score, patient_id) in enumerate(zip(top_labels, top_scores, top_patient_ids)):
                    neighbors.append((patient_id, label, float(score)))
            else:
                for j, (label, score) in enumerate(zip(top_labels, top_scores)):
                    neighbors.append((f"neighbor_{j}", label, float(score)))
            top_neighbors_list.append(neighbors)
            
            query_patient = patient_id_to_record.get(test_pid)
            if query_patient:
                query_patients.append(query_patient)
                sample_patient_ids.append(test_pid)
            else:
                logger.warning(f"Patient {test_pid} not found in manifest")
    
    logger.info(f"Processing {len(query_patients)} samples for VLM audit")
    
    # Initialize VLM Audit Module
    audit_module = VLMAuditModule(
        model_path=model_path,
        device=device,
        accept_threshold=accept_threshold,
        reject_threshold=reject_threshold,
        abstain_threshold=abstain_threshold,
    )
    
    # Get prompt template
    from .prompt_templates import get_vlm_audit_prompt
    prompt_template = get_vlm_audit_prompt(conservative=conservative)
    
    # Run audit
    audit_output_dir = output_dir / "vlm_audit"
    audit_output_dir.mkdir(parents=True, exist_ok=True)
    
    audit_results = audit_module.audit_batch(
        retrieval_predictions=retrieval_predictions,
        retrieval_similarities=retrieval_similarities,
        retrieval_labels=retrieval_labels,
        query_patients=query_patients,
        top_neighbors_list=top_neighbors_list,
        manifest=manifest,
        output_dir=audit_output_dir,
        base_dir=Path("."),
        prompt_template=prompt_template,
    )
    
    # Get ground truth labels
    ground_truth = [p.get("label") for p in query_patients]
    
    # Compute final predictions based on audit
    final_results = audit_module.compute_final_predictions(
        retrieval_predictions=retrieval_predictions,
        audit_results=audit_results["audit_results"],
        ground_truth=ground_truth,
    )
    
    # Compute audit metrics
    audit_decisions = [r["final_decision"] for r in audit_results["audit_results"]]
    audit_confidences = [r["vlm_confidence"] for r in audit_results["audit_results"]]
    
    # Compute error scores for error detection (higher = more likely error)
    # Reject decisions with high confidence = high error likelihood
    # Accept decisions with low confidence = medium error likelihood
    audit_error_scores = []
    for decision, conf in zip(audit_decisions, audit_confidences):
        if decision == "reject":
            audit_error_scores.append(conf)
        elif decision == "accept":
            audit_error_scores.append(1.0 - conf)
        else:  # abstain
            audit_error_scores.append(0.5)
    
    metrics = compute_audit_metrics(
        main_predictions=retrieval_predictions,
        ground_truth=ground_truth,
        audit_decisions=audit_decisions,
        audit_confidences=audit_confidences,
        audit_error_scores=audit_error_scores,
    )
    
    # Compile results
    results = {
        "sample_patient_ids": sample_patient_ids,
        "retrieval_predictions": retrieval_predictions,
        "ground_truth": ground_truth,
        "audit_results": audit_results["audit_results"],
        "audit_summary": {
            "accept_count": audit_results["accept_count"],
            "reject_count": audit_results["reject_count"],
            "abstain_count": audit_results["abstain_count"],
            "accept_ratio": audit_results["accept_ratio"],
            "reject_ratio": audit_results["reject_ratio"],
            "abstain_ratio": audit_results["abstain_ratio"],
        },
        "final_predictions": final_results["final_predictions"],
        "needs_human_review": final_results["needs_human_review"],
        "coverage": final_results["coverage"],
        "human_review_ratio": final_results["human_review_ratio"],
        "metrics": metrics,
    }
    
    if "accuracy_on_accepted" in final_results:
        results["accuracy_on_accepted"] = final_results["accuracy_on_accepted"]
        results["n_accepted"] = final_results["n_accepted"]
    
    # Log summary
    logger.info("=" * 60)
    logger.info("VLM Audit Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {len(retrieval_predictions)}")
    logger.info(f"Accept: {audit_results['accept_count']} ({audit_results['accept_ratio']*100:.1f}%)")
    logger.info(f"Reject: {audit_results['reject_count']} ({audit_results['reject_ratio']*100:.1f}%)")
    logger.info(f"Abstain: {audit_results['abstain_count']} ({audit_results['abstain_ratio']*100:.1f}%)")
    logger.info(f"Coverage (Auto): {final_results['coverage']*100:.1f}%")
    logger.info(f"Human Review Needed: {final_results['human_review_ratio']*100:.1f}%")
    if "accuracy_on_accepted" in final_results:
        logger.info(f"Accuracy on Accepted: {final_results['accuracy_on_accepted']*100:.1f}%")
    logger.info("-" * 60)
    logger.info("Audit Quality Metrics:")
    logger.info(f"  AUROC (Error Detection): {metrics['error_detection']['auroc']:.4f}")
    logger.info(f"  Error Rejection Rate: {metrics['abstain_quality']['error_rejection_rate']*100:.1f}%")
    logger.info(f"  Correct Rejection Rate: {metrics['abstain_quality']['correct_rejection_rate']*100:.1f}%")
    logger.info(f"  Abstain F1: {metrics['abstain_quality']['abstain_f1']:.4f}")
    logger.info(f"  AURC: {metrics['selective_prediction']['aurc']:.4f}")
    logger.info("=" * 60)
    
    # Save results
    summary_file = output_dir / "vlm_audit_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Audit summary saved to {summary_file}")
    
    # Save detailed per-sample results
    details_file = output_dir / "vlm_audit_details.jsonl"
    with open(details_file, 'w', encoding='utf-8') as f:
        for i, (pid, pred, gt, audit) in enumerate(zip(
            sample_patient_ids, retrieval_predictions, ground_truth, audit_results["audit_results"]
        )):
            record = {
                "patient_id": pid,
                "retrieval_prediction": pred,
                "ground_truth": gt,
                "retrieval_correct": pred == gt,
                "audit_decision": audit["final_decision"],
                "audit_confidence": audit["vlm_confidence"],
                "cited_neighbors": audit.get("cited_neighbors", []),
                "rationale": audit.get("rationale", ""),
                "needs_human_review": final_results["needs_human_review"][i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Detailed results saved to {details_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM Audit Pipeline for Retrieval Results"
    )
    parser.add_argument(
        "--exp_dir", 
        type=str, 
        default="outputs/experiments/example_exp",
        help="Path to experiment directory containing retrieval results"
    )
    parser.add_argument(
        "--manifest_path", 
        type=str, 
        default="data/processed/manifest.jsonl",
        help="Path to manifest file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/vlm_audit",
        help="Path to output directory"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=30,
        help="Maximum number of samples to process per fold"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to VLM model"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run VLM on"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for sampling cases"
    )
    parser.add_argument(
        "--accept_threshold", 
        type=float, 
        default=0.7,
        help="Confidence threshold for accepting prediction"
    )
    parser.add_argument(
        "--reject_threshold", 
        type=float, 
        default=0.7,
        help="Confidence threshold for rejecting prediction"
    )
    parser.add_argument(
        "--abstain_threshold", 
        type=float, 
        default=0.5,
        help="Confidence threshold for abstaining"
    )
    parser.add_argument(
        "--conservative", 
        action="store_true",
        help="Use conservative prompt (more abstain decisions)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Using experiment directory: {args.exp_dir}")
    config_path = Path(args.exp_dir) / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Experiment config: {config}")
    else:
        logger.warning(f"Config file not found at {config_path}")
    
    run_vlm_audit_pipeline(
        exp_dir=Path(args.exp_dir),
        manifest_path=Path(args.manifest_path),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        model_path=args.model_path,
        device=args.device,
        seed=args.seed,
        accept_threshold=args.accept_threshold,
        reject_threshold=args.reject_threshold,
        abstain_threshold=args.abstain_threshold,
        conservative=args.conservative,
    )


if __name__ == "__main__":
    main()
