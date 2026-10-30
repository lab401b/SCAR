# SCAR: Spectrum-based Consistency Auditing and Recovery

Official implementation accompanying the paper **"Spectrum-Based Defense Against Dynamic Backdoors in Personalized Federated Learning."**

## Overview

This repository provides the implementation of **SCAR (Spectrum-based Consistency Auditing and Recovery)**, an inference-time defense against input-adaptive backdoor attacks in Personalized Federated Learning (PFL).

SCAR consists of three phases:

1. **Spectrum-based Filtering (SF):** Uses JPEG compression to attenuate high-frequency trigger perturbations.
2. **Consistency-based Auditing (CBA):** Detects potentially triggered inputs by measuring the cosine similarity between model logits before and after filtering.
3. **Adaptive Recovery (AR):** Recovers predictions through multi-quality fusion and prediction masking.

SCAR operates on the client side without modifying or fine-tuning the trained personalized models.

## Requirements

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-compatible GPU is recommended for training and evaluation.

## Datasets and Models

The experiments include the following datasets:

- CIFAR-10
- CIFAR-100
- TinyImageNet

Supported model architectures include ResNet, MobileNet, and DenseNet.

The repository also includes implementations of several FL/PFL algorithms, along with the Bad-PFL attack and dynamic trigger generation.

Datasets and pretrained model checkpoints are not included in this repository.

## Training

The main training entry point is `main.py`.

For example, the default CIFAR-10 experiment can be configured using:

```bash
python main.py --dataset cifar10 --model resnet10 --pfl fedbn --client_num 100 --bad_client_num 10 --ba_poison_rate 0.2 --total_round 1000
```

Additional experiment configurations can be specified through the command-line arguments or the `run_experiments.py` script.

## SCAR Evaluation

The main evaluation script is:

`batch_evaluate_fed_and_detection_Cos_Similarity.py`

The evaluation measures:

- Benign Accuracy (BA)
- Attack Success Rate (ASR)
- Robust Accuracy (RA)
- Detection Accuracy
- Precision, Recall, and F1-score

For each client, benign test queries and their corresponding triggered versions are evaluated using the client's personalized model. Results are summarized across clients.

The `SCAR_overhead.py` script is provided for measuring inference latency and memory overhead.

## Additional Analysis

The repository also includes scripts for frequency-domain trigger analysis and dataset preparation.

## Reproducibility Notes

Experiment results may vary due to randomized client data partitioning and training. The original training implementation does not fully seed NumPy-based data partitioning, so exact reproduction of previously reported numerical results is not guaranteed.

Please verify the experiment parameters and local dataset and checkpoint paths before running the scripts.

## Appendix

An extended appendix containing additional experimental results and supporting analysis will be made available in this repository.

## Citation

Citation information will be added upon publication.