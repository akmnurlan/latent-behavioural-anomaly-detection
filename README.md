# Latent Behavioural Anomaly Detection (PhD prep project)

Research-oriented implementation of **semi-supervised intrusion/anomaly detection** under:
- **extreme class imbalance** (e.g., <0.1% anomalies),
- **behavioural drift / evolving attacks**, and
- the need for **early detection** (detection delay).

The project models intrusion detection as **behavioural regime change detection** rather than only per-flow classification.

## Methods (planned + in progress)
Baseline models:
- **Supervised baseline:** XGBoost / tree-based classifier.
- **Unsupervised baseline:** Autoencoder trained on normal traffic (reconstruction error).

Proposed method:
- **Latent-space change detection:** learn embeddings on normal traffic and apply change detection
  (e.g., CUSUM / BOCPD / rolling-window drift tests) to identify **behavioural regime transitions**.

## Datasets
Planned evaluation on public intrusion datasets (choose one as primary):
- CIC-IDS2017 (recommended for first results)
- UNSW-NB15

> Note: datasets are not committed to the repo. See `data/README.md` for download + preprocessing steps.

## Metrics
- Precision / Recall / F1
- False Positive Rate (FPR)
- **Detection delay** (time/flows between attack onset and detection)
- Robustness under simulated imbalance (1%, 0.5%, 0.1%)

## Reproducibility
- Fixed random seeds
- Configuration files in `configs/`
- Results exported to `results/figures/` and `results/tables/`

## Quick start
```bash
pip install -r requirements.txt
python -m src.data.make_dataset --config configs/preprocess.yaml
python -m src.experiments.run_all --config configs/experiment.yaml
