# Dataset Instructions – CIC-IDS2017

This project uses the CIC-IDS2017 dataset.

Official source:
https://www.unb.ca/cic/datasets/ids-2017.html

## Steps

1. Download CSV files from the official website.
2. Extract them into this directory (do **not** commit the dataset to GitHub).
3. Run preprocessing:

```bash
python -m src.data.make_dataset
