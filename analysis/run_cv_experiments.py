import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from config import BaseConfig
from config.encoder_config import EncoderConfig
from data.manifest import load_manifest
from encoders import create_encoder
from utils.cv_evaluator import CVRetrievalEvaluator
from utils.vlm_review import VLMReviewModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-fold CV experiments for medical image retrieval")
    
    parser.add_argument("--manifest_path", default="data/processed/manifest.jsonl", 
                        help="Path to manifest.jsonl")
    parser.add_argument("--output_dir", default="outputs/experiments", 
                        help="Output directory for experiment results")
    
    parser.add_argument("--image_encoder", default="biomedclip", 
                        choices=["qwen3_vl_8b", "qwen3_vl_2b", "clip", "vit", "biomedclip", "dino"],
                        help="Image encoder type")
    parser.add_argument("--text_encoder", default="qwen3_vl_8b",
                        choices=["qwen3_vl_8b", "qwen3_vl_2b", "clip", "biomedclip"],
                        help="Text encoder type")
    
    parser.add_argument("--fusion", default="concat",
                        choices=["concat", "image_only", "text_only", "late"],
                        help="Fusion strategy")
    
    parser.add_argument("--pca_dim", type=int, default=96,
                        help="PCA dimension for preprocessing")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Top-K for retrieval")
    
    parser.add_argument("--w_text", type=float, default=0.5,
                        help="Text weight for late fusion (0.0-1.0)")
    
    parser.add_argument("--topk_scan", action="store_true",
                        help="Enable TopK sensitivity scan")
    parser.add_argument("--topk_list", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="List of TopK values to scan")
    
    parser.add_argument("--pca_scan", action="store_true",
                        help="Enable PCA dimension scan")
    parser.add_argument("--pca_list", type=int, nargs="+", default=[64, 96, 128],
                        help="List of PCA dimensions to scan")
    
    parser.add_argument("--text_shuffle", action="store_true",
                        help="Enable text shuffle sanity check")
    
    parser.add_argument("--late_fusion_scan", action="store_true",
                        help="Enable late fusion weight scan")
    parser.add_argument("--w_text_list", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="List of text weights for late fusion")
    
    parser.add_argument("--vlm_review", action="store_true",
                        help="Enable VLM G2 review module")
    parser.add_argument("--vlm_model_path", type=str, default=None,
                        help="Path to VLM model")
    parser.add_argument("--vlm_prompt", type=str, default=None,
                        help="Custom prompt for VLM (default: use built-in prompt)")
    
    parser.add_argument("--experiment_id", type=str, default=None,
                        help="Experiment ID for result organization")
    parser.add_argument("--device", default="cuda",
                        help="Device to use")
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding")
    
    parser.add_argument("--sample_n_per_patient", type=int, default=4,
                        help="Number of slices to sample per patient")
    
    parser.add_argument("--sampling_strategy", default="uniform",
                        choices=["uniform", "random"],
                        help="Slice sampling strategy: uniform (interval sampling) or random (seed-controlled)")
    
    parser.add_argument("--skip_encoding", action="store_true",
                        help="Skip encoding if embeddings already exist")
    parser.add_argument("--embeddings_path", default=None,
                        help="Path to pre-computed embeddings")
    
    return parser.parse_args()


def load_or_encode_embeddings(
    manifest: List[Dict],
    config: BaseConfig,
    image_encoder_type: str,
    text_encoder_type: str,
    device: str,
    batch_size: int,
    sample_n_per_patient: Optional[int] = None,
    sampling_strategy: str = "uniform",
    skip_encoding: bool = False,
    embeddings_path: Optional[str] = None,
    fusion: str = "concat",
) -> Dict[str, Dict[str, np.ndarray]]:
    embeddings_dir = Path(config.features_dir)
    
    if skip_encoding and embeddings_path:
        logger.info(f"Loading pre-computed embeddings from {embeddings_path}")
        embeddings_data = np.load(embeddings_path, allow_pickle=True)
        
        result = {}
        patient_ids = embeddings_data["patient_ids"]
        
        has_image = "image_matrix" in embeddings_data
        has_text = "text_matrix" in embeddings_data
        
        for i, patient_id in enumerate(patient_ids):
            patient_id_str = str(patient_id)
            result[patient_id_str] = {
                "image": embeddings_data["image_matrix"][i] if has_image else None,
                "text": embeddings_data["text_matrix"][i] if has_text else None,
            }
        
        return result
    
    encoder_config = EncoderConfig()
    image_embeddings = {}
    text_embeddings = {}
    
    if fusion != "text_only":
        logger.info("Encoding images...")
        encoder_kwargs = {}
        if image_encoder_type == "qwen3_vl_8b":
            encoder_kwargs["qwen3_vl_8b_config"] = encoder_config.qwen3_vl_8b_config
        elif image_encoder_type == "qwen3_vl_2b":
            encoder_kwargs["qwen3_vl_2b_config"] = encoder_config.qwen3_vl_2b_config
        elif image_encoder_type == "clip":
            encoder_kwargs["clip_config"] = encoder_config.clip_config
        elif image_encoder_type == "biomedclip":
            encoder_kwargs["biomedclip_config"] = encoder_config.biomedclip_config
        elif image_encoder_type == "vit":
            encoder_kwargs["vit_config"] = encoder_config.vit_config
        elif image_encoder_type == "dino":
            encoder_kwargs["dino_config"] = encoder_config.dino_config
        
        image_encoder = create_encoder(
            encoder_type=image_encoder_type,
            device=device,
            **encoder_kwargs,
        )
        
        for record in tqdm(manifest, desc="Encoding images"):
            patient_id = record.get("patient_id")
            slices = record.get("slices", [])
            if not slices or not patient_id:
                continue
            
            try:
                slice_embeddings: List[np.ndarray] = []
                
                if sample_n_per_patient is not None and len(slices) > sample_n_per_patient:
                    if sampling_strategy == "uniform":
                        step = len(slices) // sample_n_per_patient
                        sampled_indices = list(range(0, len(slices), step))[:sample_n_per_patient]
                        logger.debug(f"Uniform sampling: {len(slices)} -> {len(sampled_indices)} slices")
                    elif sampling_strategy == "random":
                        np.random.seed(42)
                        sampled_indices = np.random.choice(len(slices), size=sample_n_per_patient, replace=False).tolist()
                        logger.debug(f"Random sampling: {len(slices)} -> {len(sampled_indices)} slices")
                    else:
                        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
                    slices = [slices[i] for i in sampled_indices]
                    logger.debug(f"Sampled {len(slices)} slices for patient {patient_id} (original: {len(record.get('slices', []))})")
                
                for i in range(0, len(slices), batch_size):
                    batch_paths = [Path(s) for s in slices[i:i+batch_size]]
                    batch_embeddings = image_encoder.encode_images(batch_paths)
                    if isinstance(batch_embeddings, np.ndarray) and batch_embeddings.ndim == 2 and batch_embeddings.size:
                        slice_embeddings.append(batch_embeddings)
                
                if slice_embeddings:
                    all_slices = np.concatenate(slice_embeddings, axis=0)
                    image_embeddings[patient_id] = all_slices.mean(axis=0).astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to encode images for patient {patient_id}: {e}")
        
        logger.info(f"Encoded images for {len(image_embeddings)} patients")
    
    if fusion != "image_only":
        logger.info("Encoding texts...")
        encoder_kwargs = {}
        if text_encoder_type == "qwen3_vl_8b":
            encoder_kwargs["qwen3_vl_8b_config"] = encoder_config.qwen3_vl_8b_config
        elif text_encoder_type == "qwen3_vl_2b":
            encoder_kwargs["qwen3_vl_2b_config"] = encoder_config.qwen3_vl_2b_config
        elif text_encoder_type == "biomedclip":
            encoder_kwargs["biomedclip_config"] = encoder_config.biomedclip_config
        elif text_encoder_type == "clip":
            encoder_kwargs["clip_config"] = encoder_config.clip_config
        
        text_encoder = create_encoder(
            encoder_type=text_encoder_type,
            device=device,
            **encoder_kwargs,
        )
        
        for record in tqdm(manifest, desc="Encoding texts"):
            patient_id = record.get("patient_id")
            if not patient_id:
                continue
            
            meta = record.get("meta", {})
            text_parts = []
            
            if meta.get("sex"):
                text_parts.append(f"性别: {meta['sex']}")
            if meta.get("age"):
                text_parts.append(f"年龄: {meta['age']}")
            if meta.get("fever"):
                text_parts.append(f"发烧: {meta['fever']}")
            if meta.get("symptom"):
                text_parts.append(f"症状: {meta['symptom']}")
            
            if text_parts:
                text = "\n".join(text_parts)
            else:
                text = record.get("text", "")
            
            if not text:
                continue
            
            try:
                text_embedding = text_encoder.encode_text(text)
                if text_embedding is not None:
                    text_embeddings[patient_id] = np.asarray(text_embedding, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to encode text for patient {patient_id}: {e}")
        
        logger.info(f"Encoded texts for {len(text_embeddings)} patients")
    
    embeddings = {}
    if fusion == "concat" or fusion == "late":
        for patient_id in image_embeddings.keys():
            if patient_id in text_embeddings:
                embeddings[patient_id] = {
                    "image": image_embeddings[patient_id],
                    "text": text_embeddings[patient_id],
                }
    elif fusion == "image_only":
        for patient_id, img_emb in image_embeddings.items():
            embeddings[patient_id] = {
                "image": img_emb,
                "text": None,
            }
    elif fusion == "text_only":
        for patient_id, txt_emb in text_embeddings.items():
            embeddings[patient_id] = {
                "image": None,
                "text": txt_emb,
            }
    
    logger.info(f"Combined embeddings for {len(embeddings)} patients (fusion={fusion})")
    
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = embeddings_dir / "combined_embeddings.npz"
    
    patient_ids = list(embeddings.keys())
    image_dim = None
    text_dim = None
    
    for patient_id in patient_ids:
        img_emb = embeddings[patient_id]["image"]
        txt_emb = embeddings[patient_id]["text"]
        if img_emb is not None:
            image_dim = img_emb.shape[-1] if img_emb.ndim > 1 else len(img_emb)
        if txt_emb is not None:
            text_dim = txt_emb.shape[-1] if txt_emb.ndim > 1 else len(txt_emb)
    
    image_matrix = None
    text_matrix = None
    
    if image_dim is not None:
        image_matrix = np.zeros((len(patient_ids), image_dim), dtype=np.float32)
        for i, patient_id in enumerate(patient_ids):
            img_emb = embeddings[patient_id]["image"]
            if img_emb is not None:
                image_matrix[i] = img_emb
    
    if text_dim is not None:
        text_matrix = np.zeros((len(patient_ids), text_dim), dtype=np.float32)
        for i, patient_id in enumerate(patient_ids):
            txt_emb = embeddings[patient_id]["text"]
            if txt_emb is not None:
                text_matrix[i] = txt_emb
    
    patient_ids_array = np.array(patient_ids, dtype=object)
    
    save_dict = {
        "patient_ids": patient_ids_array,
    }
    if image_matrix is not None:
        save_dict["image_matrix"] = image_matrix
    if text_matrix is not None:
        save_dict["text_matrix"] = text_matrix
    
    np.savez_compressed(embeddings_path, **save_dict)
    logger.info(f"Saved combined embeddings to {embeddings_path}")
    
    return embeddings


def aggregate_embeddings(embeddings: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    aggregated = {}
    for patient_id, data in embeddings.items():
        img_emb = data["image"]
        txt_emb = data["text"]
        
        if img_emb is not None:
            if img_emb.ndim == 3:
                img_emb = img_emb.mean(axis=0)
            elif img_emb.ndim == 2:
                img_emb = img_emb.mean(axis=0)
        
        aggregated[patient_id] = {
            "image": img_emb,
            "text": txt_emb,
        }
    
    return aggregated


def run_experiment(
    args: argparse.Namespace,
    config: BaseConfig,
    experiment_id: str,
    shuffle_text: bool = False,
    enable_vlm_review: bool = False,
) -> Dict:
    logger.info(f"Running experiment: {experiment_id}")
    
    logger.info(f"Loading manifest from {args.manifest_path}")
    manifest = load_manifest(args.manifest_path)
    logger.info(f"Loaded {len(manifest)} records from manifest")
    
    embeddings = load_or_encode_embeddings(
        manifest=manifest,
        config=config,
        image_encoder_type=args.image_encoder,
        text_encoder_type=args.text_encoder,
        device=args.device,
        batch_size=args.batch_size,
        sample_n_per_patient=args.sample_n_per_patient,
        sampling_strategy=args.sampling_strategy,
        skip_encoding=args.skip_encoding,
        embeddings_path=args.embeddings_path,
        fusion=args.fusion,
    )
    
    if shuffle_text:
        logger.info("Shuffling text embeddings across patients (sanity check)")
        patient_ids = list(embeddings.keys())
        text_embeddings = [embeddings[pid]["text"] for pid in patient_ids]
        np.random.shuffle(text_embeddings)
        for pid, txt_emb in zip(patient_ids, text_embeddings):
            embeddings[pid]["text"] = txt_emb
    
    embeddings = aggregate_embeddings(embeddings)
    
    patient_ids = list(embeddings.keys())
    labels = []
    for patient_id in patient_ids:
        record = next((r for r in manifest if r.get("patient_id") == patient_id), None)
        if record:
            labels.append(record.get("label", "unknown"))
    
    logger.info(f"Patient count: {len(patient_ids)}")
    logger.info(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    evaluator = CVRetrievalEvaluator(
        cv_folds=5,
        pca_dim=args.pca_dim,
        top_k=args.top_k,
        seed=config.seed,
    )
    
    results = evaluator.run_cv(
        patient_ids=patient_ids,
        labels=labels,
        embeddings=embeddings,
        fusion=args.fusion,
        top_k_list=[1, 3, 5, args.top_k],
        w_text=args.w_text,
    )
    
    vlm_review_results = None
    if enable_vlm_review:
        logger.info("Starting VLM G2 review (limiting to first 30 samples)")
        vlm_module = VLMReviewModule(
            model_path=args.vlm_model_path,
            device=args.device,
        )
        
        output_dir = Path(args.output_dir) / f"exp_{experiment_id}"
        vlm_output_dir = output_dir / "vlm_review"
        
        query_patients = [next((r for r in manifest if r.get("patient_id") == pid), None) 
                        for pid in patient_ids]
        
        retrieval_predictions = []
        retrieval_similarities = []
        retrieval_labels = []
        top_neighbors_list = []
        
        max_vlm_samples = 30
        sample_count = 0
        
        for fold_result in results["fold_results"]:
            fold_idx = fold_result.get("fold", 0)
            test_patient_ids = fold_result.get("test_patient_ids", [])
            train_ids = fold_result.get("train_ids", [])
            all_top_labels = fold_result.get("all_top_labels", [])
            all_top_scores = fold_result.get("all_top_scores", [])
            
            for i, test_pid in enumerate(test_patient_ids):
                if sample_count >= max_vlm_samples:
                    break
                
                if i < len(all_top_labels):
                    top_labels = all_top_labels[i]
                    top_scores = all_top_scores[i]
                    
                    from collections import Counter
                    vote_pred = Counter(top_labels).most_common(1)[0][0]
                    retrieval_predictions.append(vote_pred)
                    
                    retrieval_similarities.append(top_scores)
                    retrieval_labels.append(top_labels)
                    
                    neighbors = []
                    for j, (label, score) in enumerate(zip(top_labels, top_scores)):
                        neighbors.append((f"neighbor_{j}", label, float(score)))
                    top_neighbors_list.append(neighbors)
                    
                    sample_count += 1
                else:
                    retrieval_predictions.append(None)
                    retrieval_similarities.append([])
                    retrieval_labels.append([])
                    top_neighbors_list.append([])
            
            if sample_count >= max_vlm_samples:
                break
        
        logger.info(f"Processing {sample_count} samples for VLM review")
        
        vlm_review_results = vlm_module.g2_goalkeeper_review(
            retrieval_predictions=retrieval_predictions,
            retrieval_similarities=retrieval_similarities,
            retrieval_labels=retrieval_labels,
            query_patients=query_patients[:sample_count],
            top_neighbors_list=top_neighbors_list,
            manifest=manifest,
            output_dir=vlm_output_dir,
            prompt_template=args.vlm_prompt,
        )
    
    config_dict = {
        "experiment_id": experiment_id,
        "image_encoder": args.image_encoder,
        "text_encoder": args.text_encoder,
        "fusion": args.fusion,
        "pca_dim": args.pca_dim,
        "top_k": args.top_k,
        "w_text": args.w_text if args.fusion == "late" else None,
        "cv_folds": 5,
        "seed": config.seed,
        "device": args.device,
        "num_patients": len(patient_ids),
        "label_distribution": {
            str(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        },
        "text_shuffle": shuffle_text,
        "vlm_review": enable_vlm_review,
    }
    
    if vlm_review_results:
        results["vlm_review"] = vlm_review_results
    
    evaluator.save_results(
        results=results,
        output_dir=Path(args.output_dir),
        experiment_id=experiment_id,
        config=config_dict,
    )
    
    logger.info(f"Experiment {experiment_id} completed")
    logger.info(f"Summary: Top1={results['summary']['top1']['mean']:.4f}±{results['summary']['top1']['std']:.4f}, "
                f"Vote Acc={results['summary']['vote_acc']['mean']:.4f}±{results['summary']['vote_acc']['std']:.4f}")
    
    if vlm_review_results:
        logger.info(f"VLM Review: agreement_rate={vlm_review_results['agreement_rate']:.4f}, "
                   f"needs_review_ratio={vlm_review_results['needs_review_ratio']:.4f}")
    
    return results


def run_experiments(args: argparse.Namespace, config: BaseConfig):
    experiment_configs = [
        {
            "id": "exp_a_baseline",
            "fusion": "concat",
            "image_encoder": "biomedclip",
            "text_encoder": "qwen3_vl_8b",
        },
        {
            "id": "exp_b_image_encoders",
            "fusion": "concat",
            "image_encoder": "biomedclip",
            "text_encoder": "qwen3_vl_8b",
        },
        {
            "id": "exp_c_fusion_strategies",
            "fusion": "image_only",
            "image_encoder": "biomedclip",
            "text_encoder": "qwen3_vl_8b",
        },
        {
            "id": "exp_d_pca_dimensions",
            "fusion": "concat",
            "image_encoder": "biomedclip",
            "text_encoder": "qwen3_vl_8b",
        },
    ]
    
    all_results = {}
    
    for exp_config in experiment_configs:
        original_args = vars(args).copy()
        
        args.image_encoder = exp_config["image_encoder"]
        args.text_encoder = exp_config["text_encoder"]
        args.fusion = exp_config["fusion"]
        
        if exp_config["id"] == "exp_d_pca_dimensions":
            pca_dims = [64, 96, 128]
            for dim in pca_dims:
                args.pca_dim = dim
                exp_id = f"{exp_config['id']}_dim{dim}"
                results = run_experiment(args, config, exp_id)
                all_results[exp_id] = results
        else:
            exp_id = exp_config["id"]
            results = run_experiment(args, config, exp_id)
            all_results[exp_id] = results
        
        vars(args).update(original_args)
    
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "all_experiments_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"All experiments summary saved to {summary_path}")


def main():
    args = parse_args()
    
    config = BaseConfig()
    
    if args.experiment_id:
        if args.late_fusion_scan:
            logger.info(f"Running late fusion weight scan for {args.experiment_id}")
            all_results = {}
            for w_text in args.w_text_list:
                args.w_text = w_text
                exp_id = f"{args.experiment_id}_w{w_text:.2f}"
                results = run_experiment(args, config, exp_id)
                all_results[exp_id] = results
            
            output_dir = Path(args.output_dir)
            summary_path = output_dir / f"{args.experiment_id}_late_fusion_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Late fusion scan summary saved to {summary_path}")
        elif args.topk_scan:
            logger.info(f"Running TopK sensitivity scan for {args.experiment_id}")
            all_results = {}
            for top_k in args.topk_list:
                args.top_k = top_k
                exp_id = f"{args.experiment_id}_topk{top_k}"
                results = run_experiment(args, config, exp_id)
                all_results[exp_id] = results
            
            output_dir = Path(args.output_dir)
            summary_path = output_dir / f"{args.experiment_id}_topk_scan_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"TopK scan summary saved to {summary_path}")
        elif args.pca_scan:
            logger.info(f"Running PCA dimension scan for {args.experiment_id}")
            all_results = {}
            for pca_dim in args.pca_list:
                args.pca_dim = pca_dim
                exp_id = f"{args.experiment_id}_pca{pca_dim}"
                results = run_experiment(args, config, exp_id)
                all_results[exp_id] = results
            
            output_dir = Path(args.output_dir)
            summary_path = output_dir / f"{args.experiment_id}_pca_scan_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"PCA scan summary saved to {summary_path}")
        elif args.text_shuffle:
            logger.info(f"Running text shuffle sanity check for {args.experiment_id}")
            results_original = run_experiment(args, config, f"{args.experiment_id}_original")
            results_shuffled = run_experiment(args, config, f"{args.experiment_id}_shuffled", shuffle_text=True)
            
            output_dir = Path(args.output_dir)
            summary_path = output_dir / f"{args.experiment_id}_text_shuffle_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "original": results_original,
                    "shuffled": results_shuffled,
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Text shuffle summary saved to {summary_path}")
        elif args.vlm_review:
            logger.info(f"Running VLM G2 review for {args.experiment_id}")
            run_experiment(args, config, args.experiment_id, enable_vlm_review=True)
        else:
            run_experiment(args, config, args.experiment_id)
    else:
        run_experiments(args, config)


if __name__ == "__main__":
    main()
