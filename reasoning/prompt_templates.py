from typing import Dict, List, Optional


def build_diagnosis_prompt(
    patient_info: Dict,
    retrieved_cases: Optional[List[Dict]] = None,
    include_image: bool = True,
) -> str:
    prompt_parts = []
    
    prompt_parts.append("你是一位专业的放射科医生，请根据以下信息确认患者的诊断。\n")
    
    prompt_parts.append("\n## 患者信息\n")
    sex = patient_info.get("sex", patient_info.get("gender", "未知"))
    age = patient_info.get("age", "未知")
    fever = patient_info.get("fever", "未知")
    cough = patient_info.get("cough", "未知")
    symptom = patient_info.get("symptom", "未知")
    
    prompt_parts.append(f"- 性别: {sex}\n")
    prompt_parts.append(f"- 年龄: {age}\n")
    prompt_parts.append(f"- 发热: {fever}\n")
    prompt_parts.append(f"- 咳嗽: {cough}\n")
    prompt_parts.append(f"- 症状: {symptom}\n")
    
    if retrieved_cases:
        prompt_parts.append("\n## 相似病例参考\n")
        for i, case in enumerate(retrieved_cases[:5], 1):
            prompt_parts.append(f"### 病例 {i}\n")
            prompt_parts.append(f"- 诊断结果: {case.get('label', '未知')}\n")
            if 'score' in case:
                prompt_parts.append(f"- 相似度: {case['score']:.4f}\n")
            if 'text' in case:
                prompt_parts.append(f"- 文本信息: {case['text']}\n")
            prompt_parts.append("\n")
    
    prompt_parts.append("\n## 任务\n")
    prompt_parts.append("请根据患者信息、CT图像和相似病例，确认患者的诊断（以JSON格式返回）：\n")
    prompt_parts.append("```json\n")
    prompt_parts.append('{\n')
    prompt_parts.append('  "diagnosis": "诊断结果（病毒性肺炎/细菌性肺炎/PJP肺炎/正常）",\n')
    prompt_parts.append('  "confidence": "置信度（高/中/低）",\n')
    prompt_parts.append('  "reasoning": "诊断依据和推理过程",\n')
    prompt_parts.append('  "key_findings": ["关键发现1", "关键发现2", ...]\n')
    prompt_parts.append('}\n')
    prompt_parts.append("```\n")
    
    return "".join(prompt_parts)


def build_review_prompt(
    patient_info: Dict,
    retrieval_result: Dict,
    include_image: bool = True,
) -> str:
    prompt_parts = []
    
    prompt_parts.append("你是一位专业的放射科医生，请复核以下诊断结果。\n")
    
    prompt_parts.append("\n## 患者信息\n")
    sex = patient_info.get("sex", patient_info.get("gender", "未知"))
    age = patient_info.get("age", "未知")
    fever = patient_info.get("fever", "未知")
    cough = patient_info.get("cough", "未知")
    symptom = patient_info.get("symptom", "未知")
    
    prompt_parts.append(f"- 性别: {sex}\n")
    prompt_parts.append(f"- 年龄: {age}\n")
    prompt_parts.append(f"- 发热: {fever}\n")
    prompt_parts.append(f"- 咳嗽: {cough}\n")
    prompt_parts.append(f"- 症状: {symptom}\n")
    
    prompt_parts.append("\n## 检索结果\n")
    top_k = retrieval_result.get("top_k", [])
    for i, case in enumerate(top_k[:5], 1):
        prompt_parts.append(f"### Top {i}\n")
        prompt_parts.append(f"- 诊断: {case.get('label', '未知')}\n")
        prompt_parts.append(f"- 相似度: {case.get('score', 0):.4f}\n")
        if 'text' in case:
            prompt_parts.append(f"- 文本: {case['text']}\n")
        prompt_parts.append("\n")
    
    prompt_parts.append("\n## 任务\n")
    prompt_parts.append("请复核检索结果，给出最终诊断（以JSON格式返回）：\n")
    prompt_parts.append("```json\n")
    prompt_parts.append('{\n')
    prompt_parts.append('  "final_diagnosis": "最终诊断结果",\n')
    prompt_parts.append('  "confidence": "置信度（高/中/低）",\n')
    prompt_parts.append('  "agreement_with_retrieval": "是否与检索结果一致（是/否）",\n')
    prompt_parts.append('  "reasoning": "诊断依据和推理过程"\n')
    prompt_parts.append('}\n')
    prompt_parts.append("```\n")
    
    return "".join(prompt_parts)


def build_treatment_prompt(
    patient_info: Dict,
    diagnosis: str,
    include_image: bool = True,
) -> str:
    prompt_parts = []
    
    prompt_parts.append("你是一位专业的放射科医生，请根据诊断结果给出治疗建议。\n")
    
    prompt_parts.append("\n## 患者信息\n")
    sex = patient_info.get("sex", patient_info.get("gender", "未知"))
    age = patient_info.get("age", "未知")
    fever = patient_info.get("fever", "未知")
    cough = patient_info.get("cough", "未知")
    symptom = patient_info.get("symptom", "未知")
    
    prompt_parts.append(f"- 性别: {sex}\n")
    prompt_parts.append(f"- 年龄: {age}\n")
    prompt_parts.append(f"- 发热: {fever}\n")
    prompt_parts.append(f"- 咳嗽: {cough}\n")
    prompt_parts.append(f"- 症状: {symptom}\n")
    
    prompt_parts.append(f"\n## 诊断结果\n")
    prompt_parts.append(f"{diagnosis}\n")
    
    prompt_parts.append("\n## 任务\n")
    prompt_parts.append("请根据诊断结果，给出详细的治疗建议（以JSON格式返回）：\n")
    prompt_parts.append("```json\n")
    prompt_parts.append('{\n')
    prompt_parts.append('  "diagnosis_confirmation": "确认诊断结果",\n')
    prompt_parts.append('  "treatment_plan": "治疗计划概述",\n')
    prompt_parts.append('  "medications": ["药物1", "药物2", ...],\n')
    prompt_parts.append('  "recommendations": ["建议1", "建议2", ...],\n')
    prompt_parts.append('  "follow_up": "随访建议"\n')
    prompt_parts.append('}\n')
    prompt_parts.append("```\n")
    
    return "".join(prompt_parts)
