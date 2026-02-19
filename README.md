# EMR2A: Evidence-Grounded Multimodal Retrieval with Reasoning Audit for Interpretable Pneumonia Subtyping

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Official code for the EMR2A paper on interpretable pneumonia subtyping with evidence-grounded retrieval and VLM audit.

## 🎯 Overview

EMR2A is a comprehensive CT image retrieval and quality control system designed for medical diagnosis assistance. The system retrieves similar cases from a medical database using multi-modal embeddings (CT images + clinical text) and employs a VLM-based quality control layer to validate retrieval predictions.

### Key Innovation: VLM Quality Control Layer

Unlike traditional systems that use VLMs for direct diagnosis, our approach uses VLM as an **independent quality control layer**:

- **Decision Layer**: Multi-modal retrieval + majority voting determines the prediction
- **Quality Control Layer**: VLM audits the prediction without modifying labels
  - ✅ **Accept**: Prediction is well-supported by evidence → Use original prediction
  - ❌ **Reject**: Prediction contradicts evidence → Flag for human review (`NEEDS_REVIEW`)
  - ⚠️ **Abstain**: Evidence is insufficient → Flag for human review (`NEEDS_REVIEW`)

This design ensures **clinical safety** by never allowing the VLM to override the retrieval system's diagnosis, only to flag uncertain cases for expert review.

## ✨ Key Features

### Multi-Modal Retrieval

- **Image Encoding**: Support for Qwen3-VL, BioMedCLIP, ViT, CLIP, DINO encoders
- **Text Encoding**: Clinical information (symptoms, demographics) embedding
- **Fusion Strategies**: Concatenation fusion (primary) + Late fusion (ablation)
- **Similarity Search**: Cosine similarity-based Top-K retrieval

### VLM Quality Control

- **Evidence-Based Auditing**: VLM checks consistency between prediction and visual evidence
- **Neighbor Citation**: References specific Top-K cases as supporting evidence
- **Uncertainty Management**: Conservative thresholds (accept/reject ≥ 0.7, abstain < 0.5)
- **Selective Prediction**: Trade-off between coverage and accuracy

### Comprehensive Evaluation

- **5-Fold Stratified Cross-Validation**: Prevents data leakage
- **Multiple Metrics**: Top-1/3/5 accuracy, vote accuracy, macro-F1, confusion matrices
- **Error Detection**: AUROC for detecting retrieval system errors
- **Abstain Quality**: Error rejection rate, correct rejection rate

## 📁 Project Structure

```
PJP_Medical/
├── config/                         # Configuration classes
├── data/                           # Manifest loader utilities
├── encoders/                       # Image/text encoders
├── llms/                           # LLM interfaces
├── retrieval/                      # Similarity, fusion, evaluator
├── reasoning/                      # Reasoning and prompt helpers
├── pipelines/                      # Step1~Step4 pipeline entrypoints
├── baselines/                      # CNN / VLM direct baselines
├── analysis/                       # CV experiment runner
├── utils/                          # Shared utilities and metrics
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT license
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for Qwen3-VL)
- 32GB+ RAM (for large-scale retrieval)

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/Ali-Xiyao/emr2a-evidence-grounded-multimodal-retrieval.git
cd emr2a-evidence-grounded-multimodal-retrieval
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Configure model paths** (optional):

```bash
export VLM_MODEL_PATH="/path/to/qwen3-vl-8b"
export BIOMEDCLIP_PATH="/path/to/biomedclip"
```

### Running the Pipeline

#### Step 1: Build Dataset Manifest

```bash
python -m pipelines.step1_manifest.run \
    --data_root data \
    --output_dir outputs/manifest
```

#### Step 2: Generate Embeddings

```bash
python -m pipelines.step2_embeddings.run \
    --manifest_path outputs/manifest/manifest.jsonl \
    --encoder_type biomedclip \
    --output_dir outputs/embeddings
```

#### Step 3: Evaluate Retrieval

```bash
python -m pipelines.step3_retrieval.run \
    --manifest_path outputs/manifest/manifest.jsonl \
    --embeddings_path outputs/embeddings/embeddings.npz \
    --output_dir outputs/retrieval
```

#### Step 4: VLM Quality Control Audit

```bash
python -m pipelines.step4_vlm_review.run \
    --exp_dir outputs/experiments/example_exp \
    --output_dir outputs/vlm_audit \
    --max_samples 30 \
    --seed 42
```

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qwen3-VL**: Alibaba Cloud for the vision-language model
- **BioMedCLIP**: Microsoft Research for the medical CLIP model
- **Hugging Face**: For the transformers library

---

**Disclaimer**: This system is designed for research and educational purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
