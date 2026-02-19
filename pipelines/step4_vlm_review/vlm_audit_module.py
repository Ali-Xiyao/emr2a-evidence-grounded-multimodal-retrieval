"""
VLM Audit Module

This module implements VLM-based audit/quality control for retrieval results.
Unlike diagnosis prediction, the audit module acts as a "gatekeeper" that:
1. Checks consistency between retrieval prediction and visual evidence
2. Cites specific TopK neighbors as evidence
3. Abstains when evidence is weak or conflicting

Key Features:
- Audit decision: accept / reject / abstain
- Evidence citation: reference specific TopK neighbors
- Uncertainty management: abstain when confidence is low
- Selective prediction: trade-off between coverage and accuracy
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VLMAuditModule:
    """
    VLM-based audit module for retrieval results.
    
    The audit module evaluates whether the retrieval+vote prediction
    is supported by visual evidence, without making an independent diagnosis.
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        max_retries: int = 3,
        accept_threshold: float = 0.7,
        reject_threshold: float = 0.7,
        abstain_threshold: float = 0.5,
    ):
        """
        Initialize VLM Audit Module.
        
        Args:
            model_path: Path to VLM model
            device: Device to run on
            max_retries: Max retries for VLM queries
            accept_threshold: Confidence threshold for accepting prediction
            reject_threshold: Confidence threshold for rejecting prediction
            abstain_threshold: Confidence threshold for abstaining
        """
        self.model_path = model_path
        self.device = device
        self.max_retries = max_retries
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.abstain_threshold = abstain_threshold
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load VLM model and processor."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            logger.info(f"Loading VLM model from {self.model_path}")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                local_files_only=True, 
                trust_remote_code=True
            )
            self.model.eval()
            logger.info("VLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            self.model = None
            self.processor = None
    
    def create_montage(
        self,
        slice_paths: List[str],
        output_path: Optional[Path] = None,
        grid_size: Tuple[int, int] = (2, 2),
        label: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ) -> Optional[Image.Image]:
        """Create image montage from slice paths."""
        if not slice_paths:
            return None
        
        try:
            images = []
            for path in slice_paths:
                full_path = Path(path)
                if not full_path.is_absolute() and base_dir:
                    full_path = base_dir / path
                
                if not full_path.exists():
                    logger.warning(f"Image not found: {full_path}")
                    continue
                    
                img = Image.open(full_path).convert("RGB")
                images.append(img)
            
            if len(images) == 0:
                return None
            
            img_width, img_height = images[0].size
            cols, rows = grid_size
            
            montage_width = img_width * cols
            montage_height = img_height * rows
            
            montage = Image.new("RGB", (montage_width, montage_height), (255, 255, 255))
            
            for idx, img in enumerate(images):
                if idx >= cols * rows:
                    break
                row = idx // cols
                col = idx % cols
                x = col * img_width
                y = row * img_height
                montage.paste(img, (x, y))
            
            if label:
                draw = ImageDraw.Draw(montage)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                draw.text((10, 10), label, fill=(255, 0, 0), font=font)
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                montage.save(output_path)
                logger.debug(f"Saved montage to {output_path}")
            
            return montage
        except Exception as e:
            logger.error(f"Failed to create montage: {e}")
            return None
    
    def _select_even_slices(self, slice_paths: List[str], num_slices: int = 4) -> List[str]:
        """Select evenly spaced slices."""
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
    
    def construct_evidence_package(
        self,
        query_patient: Dict,
        top_neighbors: List[Tuple[str, str, float]],
        manifest: List[Dict],
        output_dir: Path,
        base_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Construct evidence package for audit.
        
        Returns package with query images, neighbor images, and metadata.
        """
        query_id = query_patient.get("patient_id")
        query_meta = query_patient.get("meta", {})
        
        # Get query images
        query_slices = query_patient.get("slices", [])
        query_images = []
        for slice_path in self._select_even_slices(query_slices, 4):
            full_path = Path(slice_path)
            if not full_path.is_absolute() and base_dir:
                full_path = base_dir / slice_path
            if full_path.exists():
                query_images.append(str(full_path))
        
        # Get neighbor images with full metadata
        neighbor_data = []
        for rank, (neighbor_id, neighbor_label, similarity) in enumerate(top_neighbors[:5], 1):
            neighbor_record = next(
                (r for r in manifest if r.get("patient_id") == neighbor_id), 
                None
            )
            if neighbor_record:
                neighbor_slices = neighbor_record.get("slices", [])
                images = []
                for slice_path in self._select_even_slices(neighbor_slices, 2):
                    full_path = Path(slice_path)
                    if not full_path.is_absolute() and base_dir:
                        full_path = base_dir / slice_path
                    if full_path.exists():
                        images.append(str(full_path))
                
                neighbor_meta = neighbor_record.get("meta", {})
                neighbor_data.append({
                    "rank": rank,
                    "id": neighbor_id,
                    "label": neighbor_label,
                    "similarity": similarity,
                    "images": images,
                    "meta": neighbor_meta,
                })
        
        # Build query text
        text_parts = []
        text_parts.append("Query Patient Information:")
        if query_meta.get("sex"):
            text_parts.append(f"- Sex: {query_meta['sex']}")
        if query_meta.get("age"):
            text_parts.append(f"- Age: {query_meta['age']}")
        if query_meta.get("fever"):
            text_parts.append(f"- Fever: {query_meta['fever']}")
        if query_meta.get("symptom"):
            text_parts.append(f"- Symptoms: {query_meta['symptom']}")
        
        query_text = "\n".join(text_parts)
        
        return {
            "query_id": query_id,
            "query_images": query_images,
            "query_text": query_text,
            "query_meta": query_meta,
            "neighbors": neighbor_data,
        }
    
    def query_vlm_audit(
        self,
        evidence_package: Dict,
        predicted_label: str,
        prompt_template: str = None,
        temperature: float = 0.0,
    ) -> Optional[Dict]:
        """
        Query VLM for audit decision.
        
        Args:
            evidence_package: Evidence package with images and metadata
            predicted_label: The retrieval+vote prediction to audit
            prompt_template: Custom prompt template
            temperature: Sampling temperature
        
        Returns:
            Dictionary with decision, confidence, cited_neighbors, rationale
        """
        if self.model is None or self.processor is None:
            logger.warning("VLM model not loaded")
            return None
        
        if prompt_template is None:
            from .prompt_templates import get_vlm_audit_prompt
            prompt_template = get_vlm_audit_prompt()
        
        try:
            # Prepare images
            query_images = evidence_package.get("query_images", [])
            if not query_images:
                logger.error("No query images found")
                return None
            
            images = []
            for img_path in query_images:
                if Path(img_path).exists():
                    images.append(Image.open(img_path).convert("RGB"))
            
            # Add top neighbor images (first 3 neighbors, 1 image each)
            neighbors = evidence_package.get("neighbors", [])
            cited_neighbors = []
            for neighbor in neighbors[:3]:
                neighbor_imgs = neighbor.get("images", [])
                if neighbor_imgs:
                    if Path(neighbor_imgs[0]).exists():
                        images.append(Image.open(neighbor_imgs[0]).convert("RGB"))
                        cited_neighbors.append(neighbor["rank"])
            
            if not images:
                logger.error("No valid images found")
                return None
            
            # Build neighbor info text
            neighbor_info = []
            for n in neighbors[:3]:
                neighbor_info.append(
                    f"  [{n['rank']}] ID: {n['id']}, Label: {n['label']}, Similarity: {n['similarity']:.3f}"
                )
            neighbor_text = "\n".join(neighbor_info)
            
            # Format prompt
            prompt = prompt_template.format(
                query_text=evidence_package["query_text"],
                predicted_label=predicted_label,
                neighbor_info=neighbor_text,
            )
            
            # Prepare messages
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate
            do_sample = temperature > 0
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=do_sample,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse output
            result = self._parse_audit_output(output_text)
            if result and cited_neighbors:
                result["cited_neighbors"] = cited_neighbors
            return result
            
        except Exception as e:
            logger.error(f"Error querying VLM: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_audit_output(self, output_text: str) -> Optional[Dict]:
        """
        Parse VLM audit output.
        
        Expected format:
        {
            "decision": "accept" | "reject" | "abstain",
            "confidence": 0.0-1.0,
            "cited_neighbors": [1, 2],
            "rationale": "brief explanation"
        }
        
        Handles Qwen3-Thinking model output which includes thinking process
        followed by </think> tag and then the actual JSON response.
        """
        try:
            output_text = output_text.strip()
            
            # For Qwen3-Thinking: JSON comes after </think> tag
            if "</think>" in output_text:
                # Extract content after </think>
                parts = output_text.split("</think>", 1)
                if len(parts) > 1:
                    json_part = parts[1].strip()
                    payload = self._extract_json_payload(json_part)
                    if payload:
                        return self._parse_audit_payload(payload)
            
            # Try to extract JSON from full text
            payload = self._extract_json_payload(output_text)
            if payload:
                return self._parse_audit_payload(payload)
            
            # Fallback: try pattern matching from thinking text
            thinking_result = self._parse_thinking_output(output_text)
            if thinking_result:
                return thinking_result
            
            # Final fallback
            return self._parse_audit_text(output_text)
            
        except Exception as e:
            logger.error(f"Error parsing audit output: {e}")
            return None
    
    def _parse_thinking_output(self, text: str) -> Optional[Dict]:
        """
        Parse Qwen3-Thinking model output.
        
        The model outputs thinking process but often includes the decision
        in the thinking text. We extract key information from it.
        """
        text_lower = text.lower()
        
        # Look for explicit decision statements in thinking
        # Patterns like: "The decision should be 'accept'", "I will choose reject", etc.
        
        decision = None
        confidence = 0.5
        cited_neighbors = []
        
        # Try to find decision
        accept_patterns = [
            r'decision\s*(should be|is|would be)\s*["\']?(accept|agree|correct)',
            r'(choose|select|pick)\s*["\']?(accept|agree)',
            r'will\s*["\']?(accept|agree)',
            r'should\s*["\']?(accept|agree)',
        ]
        reject_patterns = [
            r'decision\s*(should be|is|would be)\s*["\']?(reject|disagree|incorrect)',
            r'(choose|select|pick)\s*["\']?(reject|disagree)',
            r'will\s*["\']?(reject|disagree)',
            r'should\s*["\']?(reject|disagree)',
        ]
        abstain_patterns = [
            r'decision\s*(should be|is|would be)\s*["\']?(abstain|uncertain|unsure)',
            r'(choose|select|pick)\s*["\']?(abstain|uncertain)',
            r'will\s*["\']?(abstain|uncertain)',
            r'should\s*["\']?(abstain|uncertain)',
        ]
        
        for pattern in accept_patterns:
            if re.search(pattern, text_lower):
                decision = "accept"
                break
        
        if not decision:
            for pattern in reject_patterns:
                if re.search(pattern, text_lower):
                    decision = "reject"
                    break
        
        if not decision:
            for pattern in abstain_patterns:
                if re.search(pattern, text_lower):
                    decision = "abstain"
                    break
        
        # If no explicit decision found, try to infer from context
        if not decision:
            # Look for confidence indicators
            if "confident" in text_lower or "clear" in text_lower or "obvious" in text_lower:
                if any(word in text_lower for word in ["support", "agree", "correct", "match"]):
                    decision = "accept"
                elif any(word in text_lower for word in ["contradict", "disagree", "wrong", "mismatch"]):
                    decision = "reject"
            
            # Look for uncertainty indicators
            if not decision and any(word in text_lower for word in ["uncertain", "unclear", "ambiguous", "difficult", "hard to tell"]):
                decision = "abstain"
        
        # If still no decision, default to abstain
        if not decision:
            decision = "abstain"
        
        # Try to find confidence value
        conf_patterns = [
            r'confidence[:\s]+([0-9.]+)',
            r'([0-9.]+)\s*confidence',
            r'confidence\s*(of|is)\s*([0-9.]+)',
        ]
        for pattern in conf_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    conf_val = float(match.group(1) if match.lastindex == 1 else match.group(2))
                    if 0 <= conf_val <= 1:
                        confidence = conf_val
                        break
                    elif 0 <= conf_val <= 100:
                        confidence = conf_val / 100.0
                        break
                except:
                    pass
        
        # Try to find cited neighbors
        neighbor_patterns = [
            r'neighbor[s]?\s*(?:\[?)(\d+)(?:\]?)',
            r'case[s]?\s*(\d+)',
            r'\[(\d+)\]',
        ]
        for pattern in neighbor_patterns:
            matches = re.findall(pattern, text_lower)
            cited_neighbors.extend([int(x) for x in matches])
        
        # Remove duplicates and sort
        cited_neighbors = sorted(list(set(cited_neighbors)))
        
        return {
            "decision": decision,
            "confidence": confidence,
            "cited_neighbors": cited_neighbors,
            "rationale": text[:300],  # First 300 chars as rationale
        }
    
    def _extract_json_payload(self, raw_output: str) -> Optional[Dict]:
        """Extract JSON object from text."""
        if not raw_output:
            return None
        
        text = raw_output.strip()
        
        # Try fenced code blocks
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        
        # Try direct JSON
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except:
                try:
                    import ast
                    return ast.literal_eval(text)
                except:
                    pass
        
        # Find JSON objects
        objects = []
        depth = 0
        start = None
        in_string = False
        escape = False
        
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start:i+1])
                    start = None
        
        for candidate in reversed(objects):
            try:
                return json.loads(candidate)
            except:
                try:
                    import ast
                    return ast.literal_eval(candidate)
                except:
                    continue
        
        return None
    
    def _parse_audit_payload(self, payload: Dict) -> Optional[Dict]:
        """Parse audit decision from JSON payload."""
        # Parse decision
        decision_raw = payload.get("decision", "").lower().strip()
        if decision_raw in ["accept", "support", "agree", "yes"]:
            decision = "accept"
        elif decision_raw in ["reject", "oppose", "disagree", "no", "incorrect"]:
            decision = "reject"
        elif decision_raw in ["abstain", "uncertain", "unknown", "unsure"]:
            decision = "abstain"
        else:
            decision = "abstain"  # Default to abstain on ambiguity
        
        # Parse confidence
        confidence = payload.get("confidence", 0.5)
        if isinstance(confidence, str):
            match = re.search(r"[0-9]+(?:\.[0-9]+)?", confidence)
            confidence = float(match.group(0)) if match else 0.5
        confidence = max(0.0, min(1.0, float(confidence)))
        
        # Parse cited neighbors
        cited_neighbors = payload.get("cited_neighbors", [])
        if isinstance(cited_neighbors, str):
            # Try to parse "1, 2, 3" or "[1, 2]"
            cited_neighbors = [
                int(x.strip()) 
                for x in re.findall(r"\d+", cited_neighbors)
            ]
        
        # Parse rationale
        rationale = payload.get("rationale", "")
        
        return {
            "decision": decision,
            "confidence": confidence,
            "cited_neighbors": cited_neighbors if cited_neighbors else [],
            "rationale": rationale,
        }
    
    def _parse_audit_text(self, text: str) -> Optional[Dict]:
        """Fallback text parsing for audit output."""
        text_lower = text.lower()
        
        # Determine decision
        if any(word in text_lower for word in ["accept", "support", "agree", "correct"]):
            decision = "accept"
        elif any(word in text_lower for word in ["reject", "oppose", "disagree", "incorrect"]):
            decision = "reject"
        else:
            decision = "abstain"
        
        # Try to find confidence
        conf_match = re.search(r"confidence[:\s]+([0-9.]+)", text_lower)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        # Try to find cited neighbors
        neighbor_matches = re.findall(r"neighbor[s]?\s*(?:\[?)(\d+)(?:\]?)", text_lower)
        cited_neighbors = [int(x) for x in neighbor_matches]
        
        return {
            "decision": decision,
            "confidence": confidence,
            "cited_neighbors": cited_neighbors,
            "rationale": text[:200],  # First 200 chars as rationale
        }
    
    def audit_batch(
        self,
        retrieval_predictions: List[str],
        retrieval_similarities: List[List[float]],
        retrieval_labels: List[List[str]],
        query_patients: List[Dict],
        top_neighbors_list: List[List[Tuple[str, str, float]]],
        manifest: List[Dict],
        output_dir: Path,
        base_dir: Optional[Path] = None,
        prompt_template: str = None,
    ) -> Dict:
        """
        Run audit on a batch of retrieval predictions.
        
        Returns:
            Dictionary with audit decisions, confidences, cited neighbors, etc.
        """
        logger.info(f"Starting VLM audit on {len(query_patients)} samples")
        
        audit_results = []
        
        for idx, (query_patient, top_neighbors, pred_label) in enumerate(zip(
            query_patients, top_neighbors_list, retrieval_predictions
        )):
            logger.info(f"Auditing sample {idx+1}/{len(query_patients)}: {query_patient.get('patient_id', 'unknown')}")
            
            # Construct evidence package
            evidence_package = self.construct_evidence_package(
                query_patient, top_neighbors, manifest, output_dir, base_dir
            )
            
            # Query VLM for audit
            vlm_result = self.query_vlm_audit(
                evidence_package=evidence_package,
                predicted_label=pred_label,
                prompt_template=prompt_template,
            )
            
            if vlm_result is None:
                # Failed to get VLM output, default to abstain
                vlm_result = {
                    "decision": "abstain",
                    "confidence": 0.0,
                    "cited_neighbors": [],
                    "rationale": "VLM query failed",
                }
            
            # Apply threshold-based decision refinement
            final_decision = self._apply_decision_thresholds(
                vlm_result["decision"], 
                vlm_result["confidence"]
            )
            
            audit_results.append({
                "patient_id": query_patient.get("patient_id"),
                "predicted_label": pred_label,
                "vlm_decision": vlm_result["decision"],
                "vlm_confidence": vlm_result["confidence"],
                "final_decision": final_decision,
                "cited_neighbors": vlm_result.get("cited_neighbors", []),
                "rationale": vlm_result.get("rationale", ""),
            })
        
        # Aggregate results
        decisions = [r["final_decision"] for r in audit_results]
        accept_count = decisions.count("accept")
        reject_count = decisions.count("reject")
        abstain_count = decisions.count("abstain")
        
        logger.info(f"Audit completed:")
        logger.info(f"  Accept: {accept_count} ({accept_count/len(decisions)*100:.1f}%)")
        logger.info(f"  Reject: {reject_count} ({reject_count/len(decisions)*100:.1f}%)")
        logger.info(f"  Abstain: {abstain_count} ({abstain_count/len(decisions)*100:.1f}%)")
        
        return {
            "audit_results": audit_results,
            "accept_count": accept_count,
            "reject_count": reject_count,
            "abstain_count": abstain_count,
            "accept_ratio": accept_count / len(decisions) if decisions else 0.0,
            "reject_ratio": reject_count / len(decisions) if decisions else 0.0,
            "abstain_ratio": abstain_count / len(decisions) if decisions else 0.0,
        }
    
    def _apply_decision_thresholds(self, vlm_decision: str, vlm_confidence: float) -> str:
        """
        Apply thresholds to VLM decision.
        
        - accept: only if confidence >= accept_threshold
        - reject: only if confidence >= reject_threshold
        - abstain: if confidence < abstain_threshold or decision is uncertain
        """
        if vlm_decision == "accept":
            if vlm_confidence >= self.accept_threshold:
                return "accept"
            elif vlm_confidence < self.abstain_threshold:
                return "abstain"
            else:
                return "abstain"  # Uncertain accept
        
        elif vlm_decision == "reject":
            if vlm_confidence >= self.reject_threshold:
                return "reject"
            elif vlm_confidence < self.abstain_threshold:
                return "abstain"
            else:
                return "abstain"  # Uncertain reject
        
        else:  # abstain or unknown
            return "abstain"
    
    def compute_final_predictions(
        self,
        retrieval_predictions: List[str],
        audit_results: List[Dict],
        ground_truth: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compute final predictions based on audit decisions.
        
        Strategy:
        - accept: use retrieval prediction
        - reject: mark for human review (or use alternative if available)
        - abstain: mark for human review
        
        Returns:
            Dictionary with final predictions and coverage info
        """
        final_predictions = []
        needs_human_review = []
        
        for pred, audit in zip(retrieval_predictions, audit_results):
            decision = audit["final_decision"]
            
            if decision == "accept":
                final_predictions.append(pred)
                needs_human_review.append(False)
            elif decision == "reject":
                # For rejected samples, we could:
                # 1. Mark for human review (current approach)
                # 2. Use VLM's alternative prediction (if available)
                final_predictions.append("NEEDS_REVIEW")
                needs_human_review.append(True)
            else:  # abstain
                final_predictions.append("NEEDS_REVIEW")
                needs_human_review.append(True)
        
        # Compute coverage (accepted samples)
        coverage = sum(needs_human_review) / len(needs_human_review)
        
        result = {
            "final_predictions": final_predictions,
            "needs_human_review": needs_human_review,
            "coverage": 1.0 - coverage,  # % of samples with automated decision
            "human_review_ratio": coverage,
        }
        
        # Compute accuracy on accepted samples if ground truth available
        if ground_truth:
            accepted_indices = [i for i, needs in enumerate(needs_human_review) if not needs]
            if accepted_indices:
                accepted_preds = [retrieval_predictions[i] for i in accepted_indices]
                accepted_truth = [ground_truth[i] for i in accepted_indices]
                correct = sum(1 for p, t in zip(accepted_preds, accepted_truth) if p == t)
                result["accuracy_on_accepted"] = correct / len(accepted_indices)
                result["n_accepted"] = len(accepted_indices)
            else:
                result["accuracy_on_accepted"] = 0.0
                result["n_accepted"] = 0
        
        return result
