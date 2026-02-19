"""
VLM Audit Pipeline - Step 4

This pipeline implements VLM-based audit/quality control for retrieval results.
The audit module evaluates whether retrieval+vote predictions are supported by evidence,
without making independent diagnoses.

Key Components:
- VLMAuditModule: Main module for VLM-based audit
- AuditMetrics: Metrics for selective prediction, error detection, abstain quality
- Prompt templates: For guiding VLM audit decisions
- Evidence package: Construction of query and neighbor images with citation
"""

from .run import run_vlm_audit_pipeline
from .vlm_audit_module import VLMAuditModule
from .audit_metrics import compute_audit_metrics, AuditMetricsAggregator
from .prompt_templates import get_vlm_audit_prompt, get_vlm_diagnosis_review_prompt

__all__ = [
    "run_vlm_audit_pipeline",
    "VLMAuditModule",
    "compute_audit_metrics",
    "AuditMetricsAggregator",
    "get_vlm_audit_prompt",
    "get_vlm_diagnosis_review_prompt",
]
