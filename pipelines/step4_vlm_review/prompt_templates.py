"""
Prompt Templates for VLM Audit Module

This module provides prompt templates for VLM-based audit/quality control.
The audit module evaluates retrieval predictions without making independent diagnoses.
"""

from typing import Dict, Any


# ============================================================================
# AUDIT MODE PROMPTS (Recommended for paper)
# ============================================================================

VLM_AUDIT_PROMPT = """You are a medical AI audit system. Verify if the retrieval prediction matches the CT evidence.

{query_text}

Prediction: {predicted_label}

Similar Cases:
{neighbor_info}

Images: First 4 are query patient, rest are similar cases.

Task: Check if prediction is correct based on CT findings.

Output ONLY JSON:
{{
  "decision": "accept" | "reject" | "abstain",
  "confidence": 0.0-1.0,
  "cited_neighbors": [1, 2],
  "rationale": "brief reason"
}}

JSON:"""


VLM_AUDIT_CONSERVATIVE_PROMPT = """You are a conservative medical AI audit system. Your primary goal is to catch errors, not to maximize agreement.

{query_text}

Retrieval System Prediction: {predicted_label}

Top Retrieved Similar Cases:
{neighbor_info}

Audit Guidelines:
1. Carefully examine the CT images for patterns matching {predicted_label}
2. Compare with cited similar cases - do they truly look similar?
3. Look for contradictory evidence in the images

Decision Criteria:
- ACCEPT only if you are confident the prediction is correct (>80% sure)
- REJECT if you find clear evidence contradicting the prediction
- ABSTAIN if evidence is weak, ambiguous, or you are uncertain

Output Format (single-line JSON):
{{
  "decision": "accept" | "reject" | "abstain",
  "confidence": 0.0-1.0,
  "cited_neighbors": [1, 2, ...],
  "rationale": "brief explanation citing specific evidence"
}}

Important:
- When uncertain, always choose "abstain"
- Cite specific neighbor ranks that influenced your decision
- Output ONLY the JSON, nothing else

Return JSON now."""


VLM_AUDIT_EVIDENCE_CITATION_PROMPT = """You are an evidence-based medical audit AI. Verify predictions by citing specific evidence.

{query_text}

Prediction to Audit: {predicted_label}

Retrieved Evidence (ranked by similarity):
{neighbor_info}

Your Task:
Evaluate whether the prediction is supported by:
1. Visual patterns in the query CT images
2. Consistency with cited similar cases

Required Output (JSON):
- decision: "accept" | "reject" | "abstain"
- confidence: 0.0-1.0
- cited_neighbors: array of integers [1, 2, ...] - MUST cite at least one
- evidence_alignment: "strong" | "moderate" | "weak" | "conflicting"
- rationale: brief justification with specific citations

Decision Rules:
- accept: Strong/moderate alignment with cited evidence
- reject: Conflicting evidence or clear mismatch
- abstain: Weak alignment or insufficient evidence

Output ONLY valid JSON. No other text."""


# ============================================================================
# LEGACY PROMPTS (For backward compatibility)
# ============================================================================

VLM_DIAGNOSIS_REVIEW_PROMPT = """You are a chest CT diagnostic AI for 4-class classification.

{query_text}

Analyze the provided CT images and patient information to determine the most likely diagnosis.

You MUST output ONLY a single-line JSON object.
The first character MUST be '{{' and last character MUST be '}}'.
Output EXACTLY three keys: "diagnosis", "confidence", "rationale".
- "diagnosis" MUST be one of: "PJP", "Bacterial", "Viral", "Normal", "uncertain"
- "confidence" MUST be a number in [0.00, 1.00]
- "rationale" MUST be a brief explanation (1-2 sentences)
Do NOT output any other text, no markdown, no code fences.

Return JSON now."""


VLM_DIAGNOSIS_REVIEW_SYSTEM_MESSAGE = """You are a chest CT diagnostic AI for 4-class classification.

You MUST output ONLY a single-line JSON object with keys: "diagnosis", "confidence", "rationale".
- diagnosis: one of "PJP", "Bacterial", "Viral", "Normal", "uncertain"
- confidence: number in [0.00, 1.00]
- rationale: brief explanation
Do NOT output any other text, no markdown, no code fences."""


VLM_DIAGNOSIS_LABEL_ONLY_PROMPT = """Output EXACTLY one label from the list:
PJP, Bacterial, Viral, Normal, uncertain.
No other text, no punctuation."""


VLM_REVIEW_DECISION_PROMPT = """You are a chest CT diagnostic AI for 4-class classification.

{query_text}

Retrieval prediction: {predicted_label}

Decide whether the retrieval prediction is correct.
You MUST output ONLY a single-line JSON object.
The first character MUST be '{{' and the last character MUST be '}}'.
Output EXACTLY three keys: "decision", "confidence", "rationale".
- "decision" MUST be one of: "correct", "incorrect", "uncertain"
- "confidence" MUST be a number in [0.00, 1.00]
- "rationale" MUST be a brief explanation (1-2 sentences)
Do NOT output any other text, no markdown, no code fences.
If unsure, use "decision": "uncertain" and "confidence": 0.0.

Return JSON now."""


VLM_REVIEW_DECISION_SYSTEM_MESSAGE = """You are a chest CT diagnostic AI.

You MUST output ONLY a single-line JSON object with keys: "decision", "confidence", "rationale".
- decision: one of "correct", "incorrect", "uncertain"
- confidence: number in [0.00, 1.00]
- rationale: brief explanation
Do NOT output any other text, no markdown, no code fences."""


VLM_REVIEW_DECISION_LABEL_ONLY_PROMPT = """Output EXACTLY one word from:
correct, incorrect, uncertain.
No other text, no punctuation."""


# ============================================================================
# GETTER FUNCTIONS
# ============================================================================

def get_vlm_audit_prompt(conservative: bool = False) -> str:
    """
    Get the audit prompt template.
    
    Args:
        conservative: If True, use more conservative prompt
    
    Returns:
        Prompt template string
    """
    if conservative:
        return VLM_AUDIT_CONSERVATIVE_PROMPT
    return VLM_AUDIT_PROMPT


def get_vlm_audit_evidence_prompt() -> str:
    """Get the evidence citation audit prompt."""
    return VLM_AUDIT_EVIDENCE_CITATION_PROMPT


def get_vlm_diagnosis_review_prompt() -> str:
    """Legacy: Get diagnosis review prompt."""
    return VLM_DIAGNOSIS_REVIEW_PROMPT


def get_vlm_diagnosis_review_system_message() -> str:
    """Legacy: Get diagnosis review system message."""
    return VLM_DIAGNOSIS_REVIEW_SYSTEM_MESSAGE


def get_vlm_diagnosis_label_only_prompt() -> str:
    """Legacy: Get label-only prompt."""
    return VLM_DIAGNOSIS_LABEL_ONLY_PROMPT


def get_vlm_review_decision_prompt() -> str:
    """Legacy: Get review decision prompt."""
    return VLM_REVIEW_DECISION_PROMPT


def get_vlm_review_decision_system_message() -> str:
    """Legacy: Get review decision system message."""
    return VLM_REVIEW_DECISION_SYSTEM_MESSAGE


def get_vlm_review_decision_label_only_prompt() -> str:
    """Legacy: Get review decision label-only prompt."""
    return VLM_REVIEW_DECISION_LABEL_ONLY_PROMPT
