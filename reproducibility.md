# Reproducibility Guide

This document describes how to reproduce the experiments and results in this repository.

## Environment

Recommended:
- Python >= 3.9
- Linux/macOS (Windows should work but is not the primary target)

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Determinism & Seeds

To reduce randomness, experiments:

Set global random seeds (Python / NumPy, and Torch if used)

Use fixed data splits where applicable

Log the exact configuration for every run

Note: Full determinism is not guaranteed across different hardware/BLAS libraries, but results should be very close.

Running Experiments
1) Baseline run
```bash
python -m src.train --config configs/baseline.yaml
```
2) Change-point / drift experiment (if applicable)
```bash
python -m src.train --config configs/changepoint.yaml
```
Outputs

After running, you should see:

results/ — metrics, logs, and artifacts

reports/ — experiment summaries (if provided)

notebooks/ — analysis notebooks

Logging Format

Each run logs:

timestamp

config name / hash

model name

dataset / split information

evaluation metrics (e.g., accuracy, F1, AUROC depending on experiment)

Hardware Notes

CPU-only runs are supported for baseline experiments.

If GPU is required for deep learning components:

include GPU type (optional)

include CUDA version (optional)

Troubleshooting

If dependencies fail: upgrade pip (python -m pip install --upgrade pip)

If results differ slightly: verify the same config + seed was used

If data files are missing: check the data/ directory and README instructions


---

## Quick Fix You Should Also Do (Tiny, but big signal)

In your main README, add a link:

```markdown
📌 See [reproducibility.md](reproducibility.md) for full reproduction instructions.
