import json
import re
from typing import Dict, List, Optional, Tuple

from .prompt_templates import (
    build_diagnosis_prompt,
    build_review_prompt,
    build_treatment_prompt,
)


class ReasoningEvaluator:
    def __init__(self, llm):
        self.llm = llm
    
    def find_json_objects(self, text: str) -> List[str]:
        if not text:
            return []
        objects: List[str] = []
        depth = 0
        start: Optional[int] = None
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
                    objects.append(text[start : i + 1])
                    start = None
        return objects
    
    def extract_json_payload(self, raw_output: str) -> Optional[Dict]:
        if not raw_output:
            return None
        raw_output = raw_output.strip()
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        
        json_objects = self.find_json_objects(raw_output)
        for obj_str in json_objects:
            try:
                parsed = json.loads(obj_str)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        
        return None
    
    def diagnose(
        self,
        patient_info: Dict,
        images: Optional[List] = None,
        retrieved_cases: Optional[List[Dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Dict:
        prompt = build_diagnosis_prompt(patient_info, retrieved_cases, include_image=bool(images))
        
        if images:
            response = self.llm.generate(
                prompt=prompt,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            response = self.llm.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        
        result = self.extract_json_payload(response)
        if result is None:
            result = {
                "diagnosis": "未知",
                "confidence": "低",
                "reasoning": response,
                "key_findings": [],
            }
        
        result["raw_response"] = response
        return result
    
    def review(
        self,
        patient_info: Dict,
        retrieval_result: Dict,
        images: Optional[List] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Dict:
        prompt = build_review_prompt(patient_info, retrieval_result, include_image=bool(images))
        
        if images:
            response = self.llm.generate(
                prompt=prompt,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            response = self.llm.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        
        result = self.extract_json_payload(response)
        if result is None:
            result = {
                "final_diagnosis": "未知",
                "confidence": "低",
                "agreement_with_retrieval": "否",
                "reasoning": response,
            }
        
        result["raw_response"] = response
        return result
    
    def suggest_treatment(
        self,
        patient_info: Dict,
        diagnosis: str,
        images: Optional[List] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Dict:
        prompt = build_treatment_prompt(patient_info, diagnosis, include_image=bool(images))
        
        if images:
            response = self.llm.generate(
                prompt=prompt,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            response = self.llm.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        
        result = self.extract_json_payload(response)
        if result is None:
            result = {
                "diagnosis_confirmation": diagnosis,
                "treatment_plan": "未知",
                "medications": [],
                "recommendations": [],
                "follow_up": "未知",
            }
        
        result["raw_response"] = response
        return result
    
    def evaluate_accuracy(
        self,
        predictions: List[Dict],
        ground_truth: List[str],
    ) -> Dict[str, float]:
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        correct = 0
        for pred, gt in zip(predictions, ground_truth):
            pred_label = self._normalize_label(pred.get("diagnosis", pred.get("final_diagnosis", "")))
            gt_label = self._normalize_label(gt)
            if pred_label == gt_label:
                correct += 1
        
        accuracy = correct / len(ground_truth)
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth),
        }
    
    def _normalize_label(self, label: str) -> str:
        label = str(label).strip().lower()
        
        label_map = {
            "病毒性肺炎": "病毒性肺炎",
            "病毒性": "病毒性肺炎",
            "病毒": "病毒性肺炎",
            "viral": "病毒性肺炎",
            "细菌性肺炎": "细菌性肺炎",
            "细菌性": "细菌性肺炎",
            "细菌": "细菌性肺炎",
            "bacterial": "细菌性肺炎",
            "pjp肺炎": "pjp肺炎",
            "pjp": "pjp肺炎",
            "肺孢子菌肺炎": "pjp肺炎",
            "正常": "正常",
            "正常胸部": "正常",
            "normal": "正常",
        }
        
        for key, value in label_map.items():
            if key in label:
                return value
        
        return label
