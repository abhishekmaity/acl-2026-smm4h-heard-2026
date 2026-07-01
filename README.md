## SMM4H-HeaRD 2026 @ 64th ACL 2026
> **Reasoning Meets Evidence: LLMs for Interpretable Insomnia Detection with Evidence Extraction in Clinical Notes**

[![Paper](https://img.shields.io/badge/ACL%20Anthology-2026.smm4h--1.1-blue.svg)](https://aclanthology.org/2026.smm4h-1.1.pdf)
[![Venue](https://img.shields.io/badge/Venue-SMM4H--HeaRD%20%40%20ACL%202026-orange.svg)](https://aclanthology.org/2026.smm4h-1/)
[![Institution](https://img.shields.io/badge/C--DAC-Mumbai-green.svg)](https://www.cdac.in/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%2F%20TRL-red.svg)](https://github.com/unslothai/unsloth)

This repository contains the implementation of the systems developed by the **A3S@C-DAC** team for the **SMM4H-HeaRD 2026 Shared Task 2** (Insomnia Detection in Clinical Notes). The paper is published in the *Proceedings of the 11th Social Media Mining for Health Research and Applications (SMM4H-HeaRD) Workshop and Shared Tasks* at ACL 2026.

---

## 📑 Table of Contents
- [Abstract](#-abstract)
- [Task Description](#-task-description)
- [Methodology](#-methodology)
- [Experimental Results](#-experimental-results)
- [Error Analysis](#-error-analysis)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started](#-getting-started)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 📝 Abstract

Large language models (LLMs) have demonstrated strong capabilities in zero-shot and few-shot medical reasoning. However, their ability to align predictions with explicit clinical criteria and provide faithful, fine-grained evidence remains limited. 

Detecting insomnia from clinical narratives requires both accurate classification and clinically grounded reasoning with interpretable evidence. We present our systems for the SMM4H-HeaRD 2026 shared task, which leverages MIMIC-III notes annotated with rule-based insomnia criteria and supporting evidence spans. We explore two complementary approaches: **parameter-efficient fine-tuning (PEFT)** of lightweight models using QLoRA/LoRA, and **few-shot prompting** of larger language models for joint reasoning and evidence extraction. Our best system achieves an F1-score of **0.7333** on binary classification and a micro-F1 of **0.6535** on multi-label rule prediction, with up to **0.5192** partial-match F1 for evidence extraction. Results show that lightweight fine-tuned models can outperform larger models in classification, while larger models demonstrate stronger reasoning but struggle with precise span localization, highlighting a key gap in clinically interpretable NLP systems.

---

## 🎯 Task Description

The **SMM4H-HeaRD 2026 Shared Task 2** focuses on automatically identifying insomnia clinical status from patient records (MIMIC-III) based on specific diagnostic rules, paired with the extraction of textual evidence supporting the clinical classification.

### Subtask 1: Binary Classification
Given a clinical note, predict whether the patient is likely to have insomnia (`"yes"` or `"no"`).
* **Primary Metric:** F1-score of the `"yes"` class.

### Subtask 2: Multi-label Classification & Evidence Extraction
Predict the presence/absence of four insomnia-related criteria and extract their character-level evidence spans:
1. **Definition 1 (yes/no):** Explicit mention of insomnia by a clinician.
2. **Definition 2 (yes/no):** Indirect indicators (e.g., patient complains of fatigue or sleep issues).
3. **Rule B (yes/no):** Active prescription of hypnotic or related sleep medications.
4. **Rule C (yes/no):** General clinical rules or notes about sleep patterns.
* **Primary Metrics:** Micro-averaged F1-score for label classification, and Exact/Partial Match F1-scores for character-level evidence span extraction.

---

## 🛠 Methodology

### 1. Fine-Tuning Paradigm (Subtask 1)
* **Gemma-3-1B (QLoRA):** Built on `google/gemma-3-1b-it`. Quantized to 4-bit NormalFloat (NF4) with double quantization. Target modules include `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Trained via HuggingFace's `SFTTrainer` with a learning rate of $2 \times 10^{-4}$.
* **Qwen3-1.7B (LoRA):** Built on `unsloth/Qwen3-1.7B` using Unsloth's optimized kernel. Adapted using LoRA (rank $r=16$, alpha $\alpha=16$) targeting key attention projection matrices (`q`, `k`, `v`, `o`, `gate`, `up`, `down`). Trained with a learning rate of $1 \times 10^{-4}$.

### 2. Few-Shot Prompting Paradigm (Subtask 2)
* **Qwen3-8B (Few-Shot v1):** Large Language Model (`Qwen3-8B-Instruct`) prompted with a concise set of rule-oriented clinical descriptions, instructing it to emit a structured JSON output of labels and spans.
* **Qwen3-8B (Few-Shot v2 - Improved):** Refined prompt containing strict instructions for character span boundary alignment. Integrates a dedicated list of medication keyword triggers and prioritizes minimal, text-faithful evidence span extraction to minimize boundary truncation errors.

---

## 📊 Experimental Results

### Subtask 1: Binary Classification
Lightweight PEFT models excel at binary task adaptation, with Gemma-3-1B QLoRA obtaining perfect precision on the validation subset.

| Model | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Gemma-3-1B (QLoRA)** | **1.0000** | 0.5789 | **0.7333** |
| Qwen3-1.7B (LoRA) | 0.5217 | **0.6316** | 0.5714 |

### Subtask 2: Multi-label Rule Classification
Few-shot prompting with larger context reasoning outputs high recall. Variant `v1` achieves the highest classification micro-F1.

| Model | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Qwen3-8B (Few-Shot v1)** | **0.5789** | **0.7500** | **0.6535** |
| Qwen3-8B (Few-Shot v2) | 0.5357 | 0.6818 | 0.6000 |

### Subtask 2: Supporting Evidence Span Extraction (Micro-average)
While classification F1 dropped slightly for the `v2` prompt variant, the strict span-extraction constraints and drug keyword guidance led to a massive increase in both Exact and Partial Match F1.

| Model Variant | Exact Match F1 | Partial Match F1 |
| :--- | :---: | :---: |
| Qwen3-8B (Few-Shot v1) | 0.1395 | 0.3628 |
| **Qwen3-8B (Few-Shot v2)** | **0.4038** | **0.5192** |

---

## 🔍 Error Analysis

A qualitative evaluation of model predictions identified 5 main error categories:

| Error Category | Example Text Segment | Root Cause / Explanation |
| :--- | :--- | :--- |
| **Implicit Symptoms** | *"patient feels very tired during day"* | Model fails to link secondary clinical symptoms (fatigue, daytime lethargy) with insomnia absent explicit labels. |
| **Medication Ambiguity** | *"Lorazepam prescribed for anxiety"* | Hypnotics or sleep aids prescribed for non-insomnia conditions (anxiety, muscle spasms) trigger false-positive rule labels. |
| **Span Boundary Errors** | Correct phrase segment, incorrect offsets | Model targets correct words but introduces prefix/suffix boundary shifts, failing exact match evaluations. |
| **Negation Handling** | *"no report of sleep issues"* | LLM fails to correctly parse complex nested clinical negations. |
| **Long Context Dependency**| Evidence spread across multiple sections | Disjoint clinical remarks distributed across a lengthy document fail to be aggregated by prompt context window. |

---

## 📁 Project Directory Structure

```
.
├── DB_Report_SMM4H_HeaRD_2026_Task_2 camera-ready FINAL.pdf     # Camera-ready research paper
├── SMM4H_Insomnia_Subtask1_Gemma3_1B_QLoRA_v2/                  # Subtask 1: Gemma-3-1B QLoRA
│   ├── insomnia_gemma3_1b.ipynb                                 # Notebook: Development script
│   ├── SMM4H_Insomnia_Subtask1_Gemma3_1B_QLoRA_v2.ipynb         # Notebook: Core training script
│   ├── SMM4H_Insomnia_Subtask1_Gemma3_1B_QLoRA_v2.txt           # Score output logs
│   ├── codabench_score_v2.txt                                   # Codabench verification logs
│   ├── subtask_1_test_v2.json                                   # Generated predictions (test set)
│   └── [data files] (.json, .zip)
│
├── SMM4H_Insomnia_Subtask1_Qwen3_1dot7B_LoRA_v1/                # Subtask 1: Qwen3-1.7B LoRA
│   ├── SMM4H_Insomnia_Subtask1_Qwen3_1dot7B_LoRA_v1.ipynb       # Notebook: Training script
│   ├── SMM4H_Insomnia_Subtask1_Qwen3_1dot7B_LoRA_v1_codabench_score.txt # Score logs
│   ├── subtask_1_predictions_test.json                         # Generated predictions (test set)
│   └── [data corpus files] (.csv, .json, .zip)
│
├── SMM4H_Insomnia_Subtask2_Qwen3_8B_FewShots_v1/                 # Subtask 2: Few-Shot Prompting v1
│   ├── SMM4H_Insomnia_Subtask2_Qwen3_8B_FewShots_v1.ipynb       # Notebook: Inference pipeline
│   ├── SMM4H_Insomnia_Subtask2_Qwen3_8B_FewShots_v1_codabench_score.txt # Score logs
│   ├── subtask_2_final_submission.json                         # Final predicted rules and spans
│   └── [data corpus files] (.csv, .json, .zip)
│
└── SMM4H_Insomnia_Subtask2_Qwen3_8B_FewShots_v2/                 # Subtask 2: Few-Shot Prompting v2
    ├── SMM4H_Insomnia_Subtask2_Qwen3_8B_FewShots_v2.ipynb       # Notebook: Refined inference pipeline
    ├── SMM4H_Insomnia_Subtask2_Qwen3_8B_FewShots_v2_codabench_score.txt # Score logs
    └── [submission files] (.json, .zip)
```

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have Python 3.9+ and a GPU-enabled environment (CUDA 12+ recommended). Install the main packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl bitsandbytes accelerate datasets
```

*For Unsloth-based notebooks (e.g., Qwen3-1.7B LoRA), please install Unsloth:*
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 🏃 Running the Notebooks
1. **Subtask 1 - Gemma-3-1B QLoRA:** Navigate to `SMM4H_Insomnia_Subtask1_Gemma3_1B_QLoRA_v2/` and open `SMM4H_Insomnia_Subtask1_Gemma3_1B_QLoRA_v2.ipynb`. Run the cells sequentially to load the dataset, quantize, train, and test.
2. **Subtask 1 - Qwen3-1.7B LoRA:** Navigate to `SMM4H_Insomnia_Subtask1_Qwen3_1dot7B_LoRA_v1/` and execute `SMM4H_Insomnia_Subtask1_Qwen3_1dot7B_LoRA_v1.ipynb`.
3. **Subtask 2 - Qwen3-8B Few-Shot:** Navigate to `SMM4H_Insomnia_Subtask2_Qwen3_8B_FewShots_v2/` and open `SMM4H_Insomnia_Subtask2_Qwen3_8B_FewShots_v2.ipynb` to execute the improved span extraction pipeline.

---

## 🗂 Citation

If you use this code or refer to our findings, please cite our workshop paper:

```bibtex
@inproceedings{maity-etal-2026-a3s,
    title = "{A3S@C-DAC} at #SMM4H-HeaRD 2026: Reasoning Meets Evidence: {LLMs} for Interpretable Insomnia Detection with Evidence Extraction in Clinical Notes",
    author = "Maity, Abhishek and Shinde, Amol and Kushare, Abhishek Suresh and Pawar, Swapnil",
    booktitle = "Proceedings of the 11th Social Media Mining for Health Research and Applications (SMM4H-HeaRD 2026) Workshop and Shared Tasks",
    month = jul,
    year = "2026",
    address = "San Diego, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.smm4h-1.1.pdf",
    pages = "1--6"
}
```

---

## 🤝 Acknowledgments

We thank **Lightning AI** for generously providing the NVIDIA L4 GPU credits used in this work. We also extend our gratitude to the organizers of the **#SMM4H-HeaRD 2026** shared tasks for providing the annotated datasets.
