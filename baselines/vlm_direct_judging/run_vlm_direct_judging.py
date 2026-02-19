#!/usr/bin/env python3
"""
VLM Direct Judging Baseline

This script evaluates VLM's direct diagnostic capability without retrieval.
By default, it follows the paper protocol for Table 4:
- single run (no CV averaging)
- full cohort evaluation (all patients in manifest)
- deterministic temperature 0.0
"""

import argparse
import ast
import json
import logging
import random
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


VLM_DIRECT_JUDGING_PROMPT = """You are a chest CT diagnostic AI for 4-class classification.

## Patient Information
{patient_info}

## Task
Analyze the 4 provided chest CT images and determine the diagnosis from: PJP, Bacterial, Viral, or Normal.

## Output Format (STRICT JSON ONLY)
Output ONLY a JSON object:
{{
  "pred_label": "<PJP|Bacterial|Viral|Normal>",
  "confidence": <0.0-1.0>,
  "rationale": "<brief explanation>"
}}

Rules:
- NO thinking process
- NO step-by-step analysis  
- NO markdown code blocks
- ONLY the JSON object

JSON:"""


VLM_DIRECT_JUDGING_SYSTEM = """You are a chest CT diagnostic AI.

Output ONLY valid JSON with keys: "pred_label", "confidence", "rationale".
NO thinking, NO analysis, ONLY JSON."""


class VLMDirectJudgingBaseline:
    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        max_retries: int = 3,
    ):
        self.model_path = model_path
        self.device = device
        self.max_retries = max_retries
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            logger.info(f"Loading VLM model from {self.model_path}")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                dtype="auto",
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
            self.model.eval()
            logger.info("VLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None
    
    def _select_even_slices(self, slice_paths: List[str], num_slices: int = 4) -> List[str]:
        if not slice_paths:
            return []
        if num_slices <= 0:
            return []
        if len(slice_paths) <= num_slices:
            return list(slice_paths)
        indices = np.linspace(0, len(slice_paths) - 1, num=num_slices, dtype=int)
        selected: List[str] = []
        for idx in indices:
            path = slice_paths[int(idx)]
            if path not in selected:
                selected.append(path)
        return selected
    
    def _format_patient_info(self, patient: Dict) -> str:
        meta = patient.get("meta", {})
        parts = []
        if meta.get("sex"):
            parts.append(f"- Sex: {meta['sex']}")
        if meta.get("age"):
            parts.append(f"- Age: {meta['age']}")
        if meta.get("fever"):
            parts.append(f"- Fever: {meta['fever']}")
        if meta.get("symptom"):
            parts.append(f"- Symptoms: {meta['symptom']}")
        return "\n".join(parts) if parts else "No clinical information available"
    
    def _extract_json_payload(self, raw_output: str) -> Optional[Dict]:
        if not raw_output:
            return None
        text = raw_output.strip()

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

        json_objects = self._find_json_objects(text)
        for candidate in reversed(json_objects):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue

        pattern = r'\{\s*"pred_label"\s*:\s*"([^"]+)"\s*,\s*"confidence"\s*:\s*([0-9.]+)\s*,\s*"rationale"\s*:\s*"([^"]+)"\s*\}'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pred_label = match.group(1).strip()
            confidence = float(match.group(2))
            rationale = match.group(3).strip()
            valid_labels = ["PJP", "Bacterial", "Viral", "Normal"]
            if pred_label in valid_labels:
                return {
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "rationale": rationale,
                }

        return None

    def _find_json_objects(self, text: str) -> List[str]:
        """Find all potential JSON objects in text using brace matching."""
        results = []
        stack = []
        start_idx = -1
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == '{':
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx >= 0:
                        results.append(text[start_idx:i+1])
                        start_idx = -1

        return results
    
    def _parse_vlm_output(self, output_text: str) -> Optional[Dict]:
        """
        Parse VLM output with support for Qwen3-Thinking model.
        
        Handles:
        1. Direct JSON output
        2. JSON after </think> tag (Qwen3-Thinking)
        3. Pattern matching from thinking text
        """
        if not output_text:
            return None
        
        output_text = output_text.strip()
        
        # For Qwen3-Thinking: JSON comes after </think> tag
        if "</think>" in output_text:
            parts = output_text.split("</think>", 1)
            if len(parts) > 1:
                json_part = parts[1].strip()
                result = self._extract_json_payload(json_part)
                if result:
                    return self._validate_and_format_result(result, output_text)
        
        # Try to extract JSON from full text
        result = self._extract_json_payload(output_text)
        if result:
            return self._validate_and_format_result(result, output_text)
        
        # Fallback: try pattern matching from thinking text
        thinking_result = self._parse_thinking_output(output_text)
        if thinking_result:
            return thinking_result
        
        return None
    
    def _validate_and_format_result(self, result: Dict, raw_output: str) -> Optional[Dict]:
        """Validate and format the parsed result."""
        pred_label = result.get("pred_label", "")
        confidence = result.get("confidence", 0.0)
        rationale = result.get("rationale", "")
        
        valid_labels = ["PJP", "Bacterial", "Viral", "Normal"]
        if pred_label not in valid_labels:
            logger.warning(f"Invalid label: {pred_label}")
            return None
        
        try:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                confidence = 0.5
        except (ValueError, TypeError):
            confidence = 0.5
        
        return {
            "pred_label": pred_label,
            "confidence": confidence,
            "rationale": rationale,
            "raw_output": raw_output,
        }
    
    def _parse_thinking_output(self, text: str) -> Optional[Dict]:
        """
        Parse Qwen3-Thinking model output from thinking text.
        Extract key information when JSON parsing fails.
        """
        text_lower = text.lower()
        
        # Look for diagnosis mentions in thinking
        valid_labels = ["PJP", "Bacterial", "Viral", "Normal"]
        pred_label = None
        confidence = 0.5
        rationale = "Extracted from thinking process"
        
        # Try to find explicit diagnosis statements
        for label in valid_labels:
            # Patterns like: "diagnosis is PJP", "most likely Bacterial", etc.
            patterns = [
                rf'diagnosis\s*(is|would be|should be|appears to be)\s*["\']?{label.lower()}',
                rf'most likely\s*["\']?{label.lower()}',
                rf'(suggests?|indicates?)\s*["\']?{label.lower()}',
                rf'consistent with\s*["\']?{label.lower()}',
            ]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    pred_label = label
                    confidence = 0.6  # Lower confidence for extracted predictions
                    break
            if pred_label:
                break
        
        if pred_label:
            return {
                "pred_label": pred_label,
                "confidence": confidence,
                "rationale": rationale,
                "raw_output": text,
            }
        
        return None
    
    def predict(
        self,
        patient: Dict,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> Optional[Dict]:
        if self.model is None or self.processor is None:
            logger.warning("VLM model not loaded, returning None")
            return None
        
        if prompt_template is None:
            prompt_template = VLM_DIRECT_JUDGING_PROMPT
        if system_message is None:
            system_message = VLM_DIRECT_JUDGING_SYSTEM
        
        try:
            patient_info = self._format_patient_info(patient)
            prompt = prompt_template.format(patient_info=patient_info)
            
            slice_paths = patient.get("slices", [])
            selected_slices = self._select_even_slices(slice_paths, 4)
            
            images = []
            for img_path in selected_slices:
                full_path = Path(img_path)
                if full_path.exists():
                    images.append(Image.open(full_path).convert("RGB"))
                else:
                    logger.warning(f"Image not found: {img_path}")
            
            if not images:
                logger.error("No valid images found")
                return None
            
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "system", "content": system_message}, {"role": "user", "content": content}]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            do_sample = temperature > 0
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=temperature,
                do_sample=do_sample,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            result = self._parse_vlm_output(output_text)
            if result:
                result["patient_id"] = patient.get("patient_id")
                result["true_label"] = patient.get("label")
                result["temperature"] = temperature
            return result
        except Exception as e:
            logger.error(f"Error predicting for patient {patient.get('patient_id')}: {e}")
            import traceback
            traceback.print_exc()
            return None


def load_manifest(manifest_path: Path) -> List[Dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        if manifest_path.suffix == ".jsonl":
            import json
            records = []
            for line in f:
                records.append(json.loads(line.strip()))
            return records
        else:
            import json
            return json.load(f)


def load_fold_splits_from_experiments(
    experiment_dir: Path,
) -> Dict:
    """Load fold splits from existing experiment results."""
    splits = {}
    for fold_num in range(1, 6):
        fold_dir = experiment_dir / f"fold_{fold_num}"
        metrics_file = fold_dir / "metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                test_ids = metrics.get("test_patient_ids", [])
                splits[f"fold_{fold_num-1}"] = {
                    "test": test_ids,
                }
    
    return splits


def sample_test_cases(
    test_patient_ids: List[str],
    manifest: List[Dict],
    n_samples: int = 0,
    random_seed: int = 42,
) -> List[Dict]:
    if n_samples <= 0:
        return [r for r in manifest if r.get("patient_id") in set(test_patient_ids)]
    random.seed(random_seed)
    sampled_ids = random.sample(test_patient_ids, min(n_samples, len(test_patient_ids)))
    return [r for r in manifest if r.get("patient_id") in sampled_ids]


def evaluate_predictions(predictions: List[Dict]) -> Dict:
    if not predictions:
        return {}
    
    y_true = [p["true_label"] for p in predictions if p.get("pred_label")]
    y_pred = [p["pred_label"] for p in predictions if p.get("pred_label")]
    
    if not y_true:
        return {}
    
    accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    labels = ["PJP", "Bacterial", "Viral", "Normal"]
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, per_class_precision, per_class_recall, per_class_f1, per_class_support)
        },
        "confusion_matrix": cm.tolist(),
        "num_samples": len(y_true),
    }


def calculate_agreement(predictions_list: List[List[Dict]]) -> Dict:
    if not predictions_list or len(predictions_list) < 2:
        return {}
    
    n_runs = len(predictions_list)
    if n_runs == 0:
        return {}
    
    n_samples = len(predictions_list[0])
    if n_samples == 0:
        return {}
    
    total_agreements = 0
    total_comparisons = 0
    per_sample_agreements = []
    
    for i in range(n_samples):
        preds = [run[i]["pred_label"] for run in predictions_list if i < len(run) and run[i].get("pred_label")]
        if len(preds) < 2:
            continue
        
        most_common = Counter(preds).most_common(1)[0][0]
        agreement_count = sum(1 for p in preds if p == most_common)
        agreement_rate = agreement_count / len(preds)
        
        per_sample_agreements.append({
            "patient_id": predictions_list[0][i].get("patient_id"),
            "true_label": predictions_list[0][i].get("true_label"),
            "predictions": preds,
            "agreement_rate": float(agreement_rate),
            "most_common": most_common,
        })
        
        total_agreements += agreement_count
        total_comparisons += len(preds)
    
    overall_agreement = total_agreements / total_comparisons if total_comparisons > 0 else 0.0
    mean_agreement = np.mean([s["agreement_rate"] for s in per_sample_agreements]) if per_sample_agreements else 0.0
    
    return {
        "overall_agreement": float(overall_agreement),
        "mean_agreement": float(mean_agreement),
        "n_runs": n_runs,
        "n_samples": len(per_sample_agreements),
        "per_sample_details": per_sample_agreements,
    }


def main():
    parser = argparse.ArgumentParser(description="VLM Direct Judging Baseline")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.jsonl", help="Path to manifest file")
    parser.add_argument("--experiment_dir", type=str, default=None, help="Path to experiment directory with fold splits (only used with --fold)")
    parser.add_argument("--output_dir", type=str, default="outputs/vlm_direct_judging", help="Output directory")
    parser.add_argument("--n_samples_per_fold", type=int, default=0, help="Optional sampling size when using --fold (0 = use all)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0], help="Temperatures for inference (paper default: 0.0)")
    parser.add_argument("--model_path", type=str, default=None, help="VLM model path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--fold", type=int, default=None, help="Optional: run only one test fold (0-4) from --experiment_dir")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest(Path(args.manifest))
    
    vlm_baseline = VLMDirectJudgingBaseline(
        model_path=args.model_path,
        device=args.device,
    )
    
    all_results = {
        "config": {
            "n_samples_per_fold": args.n_samples_per_fold,
            "random_seed": args.random_seed,
            "temperatures": args.temperatures,
            "protocol": "single_run_full_cohort" if args.fold is None else "fold_based",
            "prompt_template": VLM_DIRECT_JUDGING_PROMPT,
            "system_message": VLM_DIRECT_JUDGING_SYSTEM,
        },
        "folds": {},
    }
    
    eval_sets: Dict[str, List[Dict]] = {}
    if args.fold is None:
        if args.n_samples_per_fold > 0:
            random.seed(args.random_seed)
            sampled_patients = random.sample(manifest, min(args.n_samples_per_fold, len(manifest)))
        else:
            sampled_patients = manifest
        eval_sets["all_patients"] = sampled_patients
        logger.info(f"Running paper protocol: single run on {len(sampled_patients)} patients")
    else:
        if not args.experiment_dir:
            raise ValueError("--experiment_dir is required when --fold is set")
        fold_splits = load_fold_splits_from_experiments(Path(args.experiment_dir))
        fold_key = f"fold_{args.fold}"
        fold_data = fold_splits.get(fold_key, {})
        test_ids = fold_data.get("test", [])
        if not test_ids:
            raise ValueError(f"No test samples found for {fold_key} in {args.experiment_dir}")
        sampled_patients = sample_test_cases(
            test_ids, manifest, args.n_samples_per_fold, args.random_seed
        )
        eval_sets[fold_key] = sampled_patients

    for fold_key, sampled_patients in eval_sets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {fold_key}")
        logger.info(f"{'='*60}")

        sampled_ids = [p.get("patient_id") for p in sampled_patients]
        logger.info(f"Sampled {len(sampled_patients)} patients: {sampled_ids}")
        
        fold_results = {
            "sampled_patient_ids": sampled_ids,
            "predictions": {},
            "metrics": {},
        }
        
        for temp in args.temperatures:
            temp_key = f"temp_{temp}"
            logger.info(f"Running with temperature={temp}")
            
            predictions = []
            for patient in sampled_patients:
                patient_id = patient.get("patient_id")
                result = vlm_baseline.predict(patient, temperature=temp)
                
                if result:
                    predictions.append(result)
                    logger.info(f"  {patient_id}: pred={result['pred_label']}, true={result['true_label']}, conf={result['confidence']:.2f}")
                else:
                    logger.warning(f"  {patient_id}: Failed to predict")
            
            fold_results["predictions"][temp_key] = predictions
            
            metrics = evaluate_predictions(predictions)
            logger.info(f"  Metrics: Acc={metrics.get('accuracy', 0):.3f}, F1={metrics.get('macro_f1', 0):.3f}")
            fold_results["metrics"][temp_key] = metrics
        
        if len(args.temperatures) >= 2:
            predictions_list = [fold_results["predictions"][f"temp_{t}"] for t in args.temperatures]
            agreement = calculate_agreement(predictions_list)
            fold_results["agreement"] = agreement
            logger.info(f"  Agreement: {agreement.get('overall_agreement', 0):.3f}")
        
        all_results["folds"][fold_key] = fold_results
    
    output_path = output_dir / "vlm_direct_judging_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")
    
    summary = {
        "total_samples": sum(len(f.get("sampled_patient_ids", [])) for f in all_results["folds"].values()),
        "temperatures": args.temperatures,
        "average_metrics": {},
    }
    
    for temp in args.temperatures:
        temp_key = f"temp_{temp}"
        accuracies = [f["metrics"][temp_key]["accuracy"] for f in all_results["folds"].values() if temp_key in f.get("metrics", {})]
        f1s = [f["metrics"][temp_key]["macro_f1"] for f in all_results["folds"].values() if temp_key in f.get("metrics", {})]
        
        summary["average_metrics"][temp_key] = {
            "accuracy_mean": float(np.mean(accuracies)) if accuracies else 0.0,
            "accuracy_std": float(np.std(accuracies)) if accuracies else 0.0,
            "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
            "f1_std": float(np.std(f1s)) if f1s else 0.0,
        }
    
    agreements = [f.get("agreement", {}).get("overall_agreement", 0) for f in all_results["folds"].values()]
    if agreements:
        summary["average_metrics"]["agreement"] = {
            "mean": float(np.mean(agreements)),
            "std": float(np.std(agreements)),
        }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Summary of Results:")
    logger.info("="*60)
    logger.info(f"Total samples: {summary['total_samples']}")
    for temp_key, metrics in summary["average_metrics"].items():
        if temp_key == "agreement":
            logger.info(f"{temp_key}: {metrics['mean']:.3f} ± {metrics['std']:.3f}")
        else:
            logger.info(f"{temp_key}: Acc={metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f}, F1={metrics['f1_mean']:.3f}±{metrics['f1_std']:.3f}")


if __name__ == "__main__":
    main()
