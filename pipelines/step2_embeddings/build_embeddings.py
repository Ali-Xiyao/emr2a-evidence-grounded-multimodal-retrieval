import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from config import BaseConfig
from encoders import create_encoder
from data.manifest import load_manifest


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build embeddings database from manifest")
    parser.add_argument("--manifest_path", default="outputs/manifest.jsonl", help="Path to manifest.jsonl")
    parser.add_argument("--encoder_type", default="vit", choices=["vit", "qwen3_vl", "qwen3_vl_8b", "biomedclip"], help="Encoder type")
    parser.add_argument("--model_path", default=None, help="Override model path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--output_dir", default="outputs/features", help="Output directory for embeddings")
    return parser.parse_args()


def load_images(manifest: List[Dict], image_root: Path) -> Dict[str, List[Path]]:
    image_paths = {}
    for record in manifest:
        patient_id = record.get("patient_id")
        slices = record.get("slices", [])
        if slices and patient_id:
            image_paths[patient_id] = [Path(s) for s in slices]
    return image_paths


def encode_images(encoder, image_paths: Dict[str, List[Path]], batch_size: int) -> Dict[str, np.ndarray]:
    embeddings = {}
    
    for patient_id, paths in tqdm(image_paths.items(), desc="Encoding images"):
        try:
            patient_embeddings = []
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                batch_embeddings = encoder.encode_images(batch_paths)
                patient_embeddings.append(batch_embeddings)
            
            if patient_embeddings:
                embeddings[patient_id] = np.concatenate(patient_embeddings, axis=0)
        except Exception as e:
            logger.warning(f"Failed to encode images for patient {patient_id}: {e}")
    
    return embeddings


def save_embeddings(embeddings: Dict[str, np.ndarray], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npz_path = output_dir / "embeddings.npz"
    np.savez_compressed(npz_path, **embeddings)
    logger.info(f"Saved embeddings to {npz_path}")
    
    meta_path = output_dir / "embeddings_meta.json"
    meta = {
        "num_patients": len(embeddings),
        "patients": list(embeddings.keys()),
        "embedding_dim": next(iter(embeddings.values())).shape[-1] if embeddings else 0,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metadata to {meta_path}")


def main():
    args = parse_args()
    
    logger.info(f"Loading manifest from {args.manifest_path}")
    manifest = load_manifest(args.manifest_path)
    logger.info(f"Loaded {len(manifest)} records from manifest")
    
    config = BaseConfig()
    encoder = create_encoder(
        encoder_type=args.encoder_type,
        device=args.device,
        model_path=args.model_path,
        config=config,
    )
    
    image_paths = load_images(manifest, config.data_root)
    logger.info(f"Found images for {len(image_paths)} patients")
    
    embeddings = encode_images(encoder, image_paths, args.batch_size)
    logger.info(f"Generated embeddings for {len(embeddings)} patients")
    
    save_embeddings(embeddings, Path(args.output_dir))


if __name__ == "__main__":
    main()
