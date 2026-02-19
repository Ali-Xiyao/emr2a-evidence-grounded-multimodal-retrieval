import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from config import BaseConfig
from data.manifest import load_manifest
from retrieval import RetrievalEvaluator


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument("--manifest_path", default="outputs/manifest.jsonl", help="Path to manifest.jsonl")
    parser.add_argument("--embeddings_path", default="outputs/features/embeddings.npz", help="Path to embeddings.npz")
    parser.add_argument("--output_dir", default="outputs/results", help="Output directory for results")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--text_weight", type=float, default=0.4, help="Weight for text embeddings")
    return parser.parse_args()


def load_embeddings(embeddings_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(embeddings_path)
    embeddings = {k: data[k] for k in data.files}
    return embeddings


def prepare_data(
    manifest: List[Dict],
    embeddings: Dict[str, np.ndarray],
    test_size: float = 0.2,
) -> Tuple[List[str], List[str], List[str]]:
    patient_ids = list(embeddings.keys())
    labels = []
    
    for patient_id in patient_ids:
        record = next((r for r in manifest if r.get("patient_id") == patient_id), None)
        if record:
            labels.append(record.get("label", "unknown"))
    
    train_ids, test_ids, train_labels, test_labels = train_test_split(
        patient_ids, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    return train_ids, test_ids, train_labels, test_labels


def evaluate_retrieval(
    train_ids: List[str],
    test_ids: List[str],
    train_labels: List[str],
    test_labels: List[str],
    embeddings: Dict[str, np.ndarray],
    top_k: int = 5,
) -> Dict:
    evaluator = RetrievalEvaluator()
    
    train_embeddings = np.stack([embeddings[pid].mean(axis=0) for pid in train_ids])
    test_embeddings = np.stack([embeddings[pid].mean(axis=0) for pid in test_ids])
    
    results = evaluator.evaluate_retrieval(
        train_text=None,
        test_text=None,
        train_image=train_embeddings,
        test_image=test_embeddings,
        train_labels=train_labels,
        test_labels=test_labels,
        fusion_type="none",
        top_k_list=[1, 3, 5, top_k],
    )
    
    return results


def save_results(results: Dict, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved results to {output_path}")


def main():
    args = parse_args()
    
    logger.info(f"Loading manifest from {args.manifest_path}")
    manifest = load_manifest(args.manifest_path)
    logger.info(f"Loaded {len(manifest)} records from manifest")
    
    logger.info(f"Loading embeddings from {args.embeddings_path}")
    embeddings = load_embeddings(Path(args.embeddings_path))
    logger.info(f"Loaded embeddings for {len(embeddings)} patients")
    
    train_ids, test_ids, train_labels, test_labels = prepare_data(
        manifest, embeddings, args.test_size
    )
    logger.info(f"Train set: {len(train_ids)}, Test set: {len(test_ids)}")
    
    results = evaluate_retrieval(
        train_ids, test_ids, train_labels, test_labels, embeddings, args.top_k
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "retrieval_results.json"
    save_results(results, results_path)
    
    logger.info("Retrieval evaluation completed")
    logger.info(f"Top-1 accuracy: {results.get('top1', 0):.4f}")
    logger.info(f"Top-3 accuracy: {results.get('top3', 0):.4f}")
    logger.info(f"Top-5 accuracy: {results.get('top5', 0):.4f}")


if __name__ == "__main__":
    main()
