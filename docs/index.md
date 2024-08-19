# MLPerf Inference Benchmarks

## Overview
This document provides details on various [MLPerf Inference Benchmarks](index_gh.md) categorized by tasks, models, and datasets. Each section lists the models performing similar tasks, with details on datasets, accuracy, and server latency constraints.

---

## 1. Image Classification
### [ResNet50-v1.5](benchmarks/image_classification/resnet50.md)
- **Dataset**: Imagenet-2012 (224x224) Validation
  - **Dataset Size**: 50,000
  - **QSL Size**: 1,024
- **Number of Parameters**: 25.6 Million
- **FLOPs**: TBD
- **Reference Model Accuracy**: 76.46% ACC
- **Server Scenario Latency Constraint**: 15ms
- **Equal Issue mode**: False
- **Accuracy Variants**: 99% of fp32 reference model accuracy
- **Submission Category**: Data Center, Edge

---

## 2. Text to Image
### [Stable Diffusion](benchmarks/text_to_image/sdxl.md)
- **Dataset**: Subset of Coco2014
  - **Dataset Size**: 5,000
  - **QSL Size**: 5,000
- **Number of Parameters**: 3.5 Billion <!-- taken from https://stability.ai/news/stable-diffusion-sdxl-1-announcement -->
- **FLOPs**: TBD
- **Required Accuracy (Closed Division)**:
  - FID: 23.01085758 ≤ FID ≤ 23.95007626
  - CLIP: 32.68631873 ≤ CLIP ≤ 31.81331801
- **Equal Issue mode**: False
- **Accuracy Variants**: 98% of fp32 reference model accuracy
- **Submission Category**: Data Center, Edge

---

## 3. Object Detection
### [Retinanet](benchmarks/object_detection/retinanet.md)
- **Dataset**: OpenImages
  - **Dataset Size**: 24,781
  - **QSL Size**: 64
- **Number of Parameters**: TBD
- **FLOPs**: TBD
- **Reference Model Accuracy**: 0.3755 mAP
- **Server Scenario Latency Constraint**: 100ms
- **Equal Issue mode**: False
- **Accuracy Variants**: 99% of fp32 reference model accuracy
- **Submission Category**: Data Center, Edge

---

## 4. Medical Image Segmentation
### [3d-unet](benchmarks/medical_imaging/3d-unet.md)
- **Dataset**: KiTS2019
  - **Dataset Size**: 42
  - **QSL Size**: 42
- **Number of Parameters**: 19 Million <!-- taken from https://arxiv.org/pdf/1606.06650 -->
- **FLOPs**: TBD
- **Reference Model Accuracy**: 0.86330 Mean DICE Score
- **Server Scenario**: Not Applicable
- **Equal Issue mode**: True
- **Accuracy Variants**: 99% and 99.9% of fp32 reference model accuracy
- **Submission Category**: Data Center, Edge

---

## 5. Language Tasks

### 5.1. Question Answering

### [Bert-Large](benchmarks/language/bert.md)
- **Dataset**: Squad v1.1 (384 Sequence Length)
  - **Dataset Size**: 10,833
  - **QSL Size**: 10,833
- **Number of Parameters**: 340 Million <!-- taken from https://huggingface.co/transformers/v2.9.1/pretrained_models.html -->
- **FLOPs**: TBD
- **Reference Model Accuracy**: F1 Score = 90.874%
- **Server Scenario Latency Constraint**: 130ms
- **Equal Issue mode**: False
- **Accuracy Variants**: 99% and 99.9% of fp32 reference model accuracy
- **Submission Category**: Data Center, Edge

### [LLAMA2-70B](benchmarks/language/llama2-70b.md)
- **Dataset**: OpenORCA (GPT-4 split, max_seq_len=1024)
  - **Dataset Size**: 24,576
  - **QSL Size**: 24,576
- **Number of Parameters**: 70 Billion
- **FLOPs**: TBD
- **Reference Model Accuracy**:
  - Rouge1: 44.4312
  - Rouge2: 22.0352
  - RougeL: 28.6162
  - Tokens_per_sample: 294.45
- **Server Scenario Latency Constraint**:
  - TTFT: 2000ms
  - TPOT: 200ms
- **Equal Issue mode**: True
- **Accuracy Variants**: 99% and 99.9% of fp32 reference model accuracy
- **Submission Category**: Data Center

### 5.2. Text Summarization

### [GPT-J](benchmarks/language/gpt-j.md)
- **Dataset**: CNN Daily Mail v3.0.0
  - **Dataset Size**: 13,368
  - **QSL Size**: 13,368
- **Number of Parameters**: 6 Billion
- **FLOPs**: TBD
- **Reference Model Accuracy**:
  - Rouge1: 42.9865
  - Rouge2: 20.1235
  - RougeL: 29.9881
  - Gen_len: 4,016,878
- **Server Scenario Latency Constraint**: 20s
- **Equal Issue mode**: True
- **Accuracy Variants**: 99% and 99.9% of fp32 reference model accuracy
- **Submission Category**: Data Center, Edge

### 5.3. Mixed Tasks (Question Answering, Math, and Code Generation)

### [Mixtral-8x7B](benchmarks/language/mixtral-8x7b.md)
- **Datasets**:
  - OpenORCA (5k samples of GPT-4 split, max_seq_len=2048)
  - GSM8K (5k samples of the validation split, max_seq_len=2048)
  - MBXP (5k samples of the validation split, max_seq_len=2048)
  - **Dataset Size**: 15,000
  - **QSL Size**: 15,000
- **Number of Parameters**: 47 Billion <!-- https://huggingface.co/blog/moe -->
- **FLOPs**: TBD
- **Reference Model Accuracy**:
  - Rouge1: 45.4911
  - Rouge2: 23.2829
  - RougeL: 30.3615
  - GSM8K Accuracy: 73.78%
  - MBXP Accuracy: 60.12%
  - Tokens_per_sample: 294.45
- **Server Scenario Latency Constraint**:
  - TTFT: 2000ms
  - TPOT: 200ms
- **Equal Issue mode**: True
- **Accuracy Variants**: 99% and of fp16 reference model
- **Submission Category**: Data Center

---

## 6. Recommendation
### [DLRMv2](benchmarks/recommendation/dlrm-v2.md)
- **Dataset**: Synthetic Multihot Criteo
  - **Dataset Size**: 204,800
  - **QSL Size**: 204,800
- **Number of Parameters**: TBD
- **FLOPs**: TBD
- **Reference Model Accuracy**: AUC = 80.31%
- **Server Scenario Latency Constraint**: 60ms
- **Equal Issue mode**: False
- **Accuracy Variants**: 99% and 99.9% of fp32 reference model accuracy
- **Submission Category**: Data Center

---

### Submission Categories
- **Datacenter Category**: All the benchmarks can participate.
- **Edge Category**: All benchmarks except DLRMv2, LLAMA2, and Mixtral-8x7B can participate.

### High Accuracy Variants
- **Benchmarks**: `bert`, `llama2-70b`, `dlrm_v2`, and `3d-unet`
- **Requirement**: Must achieve at least 99.9% of the reference model accuracy, compared to the default 99% accuracy requirement.