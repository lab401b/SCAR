# SCAR: Spectrum-based Consistency Auditing and Recovery

Official implementation accompanying the paper **"Spectrum-Based Defense Against Dynamic Backdoors in Personalized Federated Learning."**

## Overview

This repository provides the implementation of **SCAR (Spectrum-based Consistency Auditing and Recovery)**, an inference-time defense against input-adaptive backdoor attacks in Personalized Federated Learning (PFL).

SCAR consists of three phases:

1. **Spectrum-based Filtering (SF):** Uses JPEG compression to attenuate high-frequency trigger perturbations.
2. **Consistency-based Auditing (CBA):** Detects potentially triggered queries by measuring the cosine similarity between model logits before and after filtering.
3. **Adaptive Recovery (AR):** Recovers predictions through multi-quality fusion and prediction masking.

SCAR operates entirely on the client side without modifying or fine-tuning the deployed personalized models and does not require benign reference data.

## Supplementary Material

Additional robustness evaluations of SCAR and CBA across different datasets, model architectures, and attack configurations are provided in the supplementary material.

[**View the Supplementary Material (PDF)**](./SCAR_Supplementary_Material.pdf)

## Repository Structure

The repository is organized as follows:

```text
SCAR/
├── README.md
├── SCAR_Supplementary_Material.pdf
└── code/
    ├── main.py
    ├── client.py
    ├── server.py
    ├── run_experiments.py
    ├── batch_evaluate_fed_and_detection_Cos_Similarity.py
    ├── SCAR_overhead.py
    ├── requirements.txt
    └── ...
```

- **`code/`** contains the source code for FL/PFL training, Bad-PFL attacks, SCAR evaluation, and additional analyses.
- **`SCAR_Supplementary_Material.pdf`** provides additional robustness evaluations across datasets, model architectures, and attack configurations.
## Requirements

The implementation requires Python and the dependencies listed in `code/requirements.txt`.

From the repository root, navigate to the source code directory and install the required dependencies:

```bash
cd code
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

The main training entry point is `code/main.py`.

After navigating to the `code/` directory, the default CIFAR-10 experiment can be configured using:

```bash
python main.py --dataset cifar10 --model resnet10 --pfl fedbn --client_num 100 --bad_client_num 10 --ba_poison_rate 0.2 --total_round 1000
```

Additional experiment configurations can be specified through the command-line arguments or the `code/run_experiments.py` script.

## SCAR Evaluation

The main evaluation script is:

`code/batch_evaluate_fed_and_detection_Cos_Similarity.py`

The evaluation measures:

- Benign Accuracy (BA)
- Attack Success Rate (ASR)
- Robust Accuracy (RA)
- Detection Accuracy
- Precision, Recall, and F1-score

For each client, benign test queries and their corresponding triggered versions are evaluated using the client's personalized model. Results are summarized across clients.

The `code/SCAR_overhead.py` script is provided for measuring inference latency and memory overhead.

## Additional Analysis

The `code/` directory also includes scripts for frequency-domain trigger analysis and dataset preparation.

## Reproducibility Notes

Experiment results may vary due to randomized client data partitioning and training. The original training implementation does not fully seed NumPy-based data partitioning, so exact reproduction of previously reported numerical results is not guaranteed.

Please verify the experiment parameters and local dataset and checkpoint paths before running the scripts. Relative paths may need adjustment depending on the working directory.

## Appendix

An extended appendix containing additional experimental results and supporting analysis will be made available in the `appendix/` directory.

## Citation

Citation information will be added upon publication.
