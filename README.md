# Latent Behavioural Anomaly Detection
PhD preparation research project – Cybersecurity & Machine Learning

This repository investigates intrusion detection under:
- Extreme class imbalance (<0.1% anomalies)
- Behavioural drift
- Early detection requirements

The project formulates intrusion detection as behavioural regime change detection in latent space rather than per-flow classification.

---

## Dataset

Primary dataset:
CIC-IDS2017

See `data/README.md` for download and preprocessing instructions.

---

## Models

### Baseline 1 – Supervised
- XGBoost classifier
- Performance evaluated under simulated imbalance (1%, 0.5%, 0.1%)

### Baseline 2 – Unsupervised
- Autoencoder trained on normal traffic only
- Reconstruction error thresholding

### Proposed Method
- Latent representation learning
- Change-point detection (CUSUM / rolling window)
- Detection delay evaluation

---

## Evaluation Metrics
- Precision
- Recall
- F1-score
- False Positive Rate (FPR)
- Detection Delay

---

## Reproducibility
- Configuration-driven experiments (`configs/`)
- Fixed random seeds
- Results exported to `results/`

---

## Repository Structure
- `src/` core implementation
- `configs/` experiment settings
- `notebooks/` exploratory analysis
- `results/` figures and tables
- `reports/` technical documentation

---

## Status (Feb–Apr 2026)
- Data preprocessing
- Baseline model implementation
- Latent change detection development
- Technical report preparation
