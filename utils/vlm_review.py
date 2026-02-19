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


class VLMReviewModule:
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
            from transformers import AutoModelForVision2Seq, AutoProcessor
            logger.info(f"Loading VLM model from {self.model_path}")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=self.device,
                local_files_only=True,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
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
    
    def construct_evidence_package(
        self,
        query_patient: Dict,
        top_neighbors: List[Tuple[str, str, float]],
        manifest: List[Dict],
        output_dir: Path,
        base_dir: Optional[Path] = None,
    ) -> Dict:
        query_id = query_patient.get("patient_id")
        query_meta = query_patient.get("meta", {})
        
        query_slices = query_patient.get("slices", [])
        query_montage_path = output_dir / f"query_{query_id}_montage.png"
        self.create_montage(query_slices[:4], query_montage_path, (2, 2), f"Query Patient", base_dir)
        
        neighbor_montages = []
        for neighbor_id, neighbor_label, similarity in top_neighbors[:5]:
            neighbor_record = next((r for r in manifest if r.get("patient_id") == neighbor_id), None)
            if neighbor_record:
                neighbor_slices = neighbor_record.get("slices", [])
                neighbor_montage_path = output_dir / f"neighbor_{neighbor_id}_montage.png"
                self.create_montage(neighbor_slices[:4], neighbor_montage_path, (2, 2), f"Similar Case (sim: {similarity:.3f})", base_dir)
                neighbor_montages.append({
                    "id": neighbor_id,
                    "label": neighbor_label,
                    "similarity": similarity,
                    "montage_path": str(neighbor_montage_path),
                })
        
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
        
        text_parts.append("\nTop 5 Similar Cases (for reference):")
        for i, neighbor in enumerate(neighbor_montages, 1):
            text_parts.append(f"{i}. Label: {neighbor.get('label', 'Unknown')}, Similarity: {neighbor['similarity']:.3f}")
        
        query_text = "\n".join(text_parts)
        
        return {
            "query_id": query_id,
            "query_montage_path": str(query_montage_path),
            "query_text": query_text,
            "query_meta": query_meta,
            "neighbors": neighbor_montages,
        }
    
    def query_vlm(
        self,
        evidence_package: Dict,
        prompt_template: str = None,
    ) -> Optional[Dict]:
        if self.model is None or self.processor is None:
            logger.warning("VLM model not loaded, returning None")
            return None
        
        if prompt_template is None:
            prompt_template = """你是一位胸部CT影像诊断AI，专注于肺部感染性疾病的影像学鉴别。

【四分类标签集合】
1) 孢子虫肺炎 (PJP)
2) 细菌性肺炎 (Bacterial)
3) 病毒性肺炎 (Viral)
4) 正常 (Normal)

【输入信息】
{query_text}

【诊断要求】
- 基于CT影像特征进行诊断
- 参考相似病例的标签和相似度
- 考虑患者的年龄、性别、发烧、咳嗽等基础信息
- 给出四分类诊断（PJP/Bacterial/Viral/Normal）和置信度（0.00-1.00）

【最终输出格式】
只输出一行，格式为：诊断,置信度,覆盖状态
例如：PJP,0.85,covered
不要输出任何推理过程或其他内容。
"""
        
        try:
            query_meta = evidence_package.get("query_meta", {})
            
            age = query_meta.get("age", "未知")
            sex = query_meta.get("sex", "未知")
            fever = query_meta.get("fever", "未知")
            symptom = query_meta.get("symptom", "未知")
            
            prompt = prompt_template.format(
                年龄=age,
                性别=sex,
                是否发烧=fever,
                是否咳嗽=symptom,
                query_text=evidence_package["query_text"]
            )
            
            montage_path = evidence_package["query_montage_path"]
            if not Path(montage_path).exists():
                logger.error(f"Montage image not found: {montage_path}")
                return None
            
            image = Image.open(montage_path).convert("RGB")
            
            messages = [
                {
                    "role": "system",
                    "content": "你是医疗诊断助手。只输出一行，格式为：诊断,置信度,覆盖状态（例如：PJP,0.85,covered）。不要输出推理过程或解释。",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            result = self._parse_vlm_output(output_text)
            return result
        except Exception as e:
            logger.error(f"Failed to query VLM: {e}")
            return None
    
    def _parse_vlm_output(self, output_text: str) -> Optional[Dict]:
        try:
            import json
            
            output_text = output_text.strip()
            
            valid_classes = ["PJP", "Bacterial", "Viral", "Normal"]
            
            # First try to parse as JSON
            json_pattern = r'\{[^{}]*\}'
            json_match = re.search(json_pattern, output_text)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                    
                    diagnosis = result.get("diagnosis", "").strip()
                    confidence = result.get("confidence", 0.5)
                    coverage = result.get("coverage", "unknown")
                    
                    if diagnosis.lower() == "uncertain":
                        return {
                            "pred": "uncertain",
                            "confidence": 0.0,
                            "rationale": ["VLM marked as uncertain"]
                        }
                    
                    if diagnosis in valid_classes:
                        if confidence > 1.0:
                            confidence = confidence / 100.0
                        
                        return {
                            "pred": diagnosis,
                            "confidence": confidence,
                            "rationale": [f"Extracted from JSON, Coverage={coverage}"]
                        }
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails, try to parse the text format
            diagnosis_match = re.search(r'Diagnosis:\s*(\w+)', output_text, re.IGNORECASE)
            confidence_match = re.search(r'Confidence:\s*([0-9.]+)', output_text, re.IGNORECASE)
            coverage_match = re.search(r'Coverage:\s*(\w+)', output_text, re.IGNORECASE)
            
            if diagnosis_match:
                diagnosis = diagnosis_match.group(1).strip()
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                coverage = coverage_match.group(1).strip() if coverage_match else "unknown"
                
                if diagnosis.lower() == "uncertain":
                    return {
                        "pred": "uncertain",
                        "confidence": 0.0,
                        "rationale": ["VLM marked as uncertain"]
                    }
                
                if diagnosis in valid_classes:
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                    
                    return {
                        "pred": diagnosis,
                        "confidence": confidence,
                        "rationale": [f"Extracted from text format, Coverage={coverage}"]
                    }
            
            logger.warning(f"Failed to extract prediction from VLM output: {output_text[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Error parsing VLM output: {e}")
            return None
    
    def g2_goalkeeper_review(
        self,
        retrieval_predictions: List[str],
        retrieval_similarities: List[List[float]],
        retrieval_labels: List[List[str]],
        query_patients: List[Dict],
        top_neighbors_list: List[List[Tuple[str, str, float]]],
        manifest: List[Dict],
        output_dir: Path,
        prompt_template: str = None,
        base_dir: Optional[Path] = None,
    ) -> Dict:
        logger.info("Starting G2 goalkeeper review")
        
        vlm_predictions = []
        vlm_confidences = []
        agreements = []
        needs_review_flags = []
        
        final_predictions = []
        
        for idx, (query_patient, top_neighbors, pred_vote) in enumerate(zip(
            query_patients, top_neighbors_list, retrieval_predictions
        )):
            evidence_dir = output_dir / f"evidence_{idx}"
            evidence_package = self.construct_evidence_package(
                query_patient, top_neighbors, manifest, evidence_dir, base_dir
            )
            
            vlm_result = self.query_vlm(evidence_package, prompt_template)
            
            if vlm_result:
                vlm_pred = vlm_result["pred"]
                vlm_conf = vlm_result["confidence"]
                vlm_predictions.append(vlm_pred)
                vlm_confidences.append(vlm_conf)
                
                if vlm_pred == pred_vote:
                    agreements.append(1)
                    needs_review_flags.append(0)
                    final_predictions.append(pred_vote)
                else:
                    agreements.append(0)
                    needs_review_flags.append(1)
                    final_predictions.append(pred_vote)
            else:
                vlm_predictions.append(None)
                vlm_confidences.append(0.0)
                agreements.append(0)
                needs_review_flags.append(1)
                final_predictions.append(pred_vote)
        
        agreement_rate = np.mean(agreements) if agreements else 0.0
        needs_review_ratio = np.mean(needs_review_flags) if needs_review_flags else 0.0
        
        results = {
            "vlm_predictions": vlm_predictions,
            "vlm_confidences": vlm_confidences,
            "agreements": agreements,
            "needs_review_flags": needs_review_flags,
            "final_predictions": final_predictions,
            "agreement_rate": agreement_rate,
            "needs_review_ratio": needs_review_ratio,
        }
        
        logger.info(f"G2 review completed: agreement_rate={agreement_rate:.4f}, needs_review_ratio={needs_review_ratio:.4f}")
        
        return results
