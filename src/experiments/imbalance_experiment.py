import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

from xgboost import XGBClassifier


DATA_PATH = "data/processed/unsw_nb15_processed.csv"
LABEL_COL = "label"

def downsample_attacks(X_train, y_train, anomaly_ratio, seed=42):
    rng = np.random.default_rng(seed)

    y_arr = np.asarray(y_train)
    normal_idx = np.where(y_arr == 0)[0]
    attack_idx = np.where(y_arr == 1)[0]

    # want: attacks = ratio * normals
    n_attack_target = int(len(normal_idx) * anomaly_ratio)
    n_attack_target = max(1, min(n_attack_target, len(attack_idx)))

    attack_sample = rng.choice(attack_idx, size=n_attack_target, replace=False)
    selected = np.concatenate([normal_idx, attack_sample])
    rng.shuffle(selected)

    return X_train[selected], y_arr[selected]

def main():
    df = pd.read_csv(DATA_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected '{LABEL_COL}' in columns. Got: {df.columns}")

    X = df.drop(columns=[LABEL_COL]).values
    y = df[LABEL_COL].values

    # Keep test set stratified on original distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ratios = [0.10, 0.01, 0.005, 0.001]  # 10%, 1%, 0.5%, 0.1%
    f1s, ps, rs = [], [], []

    for r in ratios:
        X_tr, y_tr = downsample_attacks(X_train, y_train, anomaly_ratio=r, seed=42)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        p = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        f1s.append(f1); ps.append(p); rs.append(rec)
        print(f"ratio={r:>6} | F1={f1:.4f} | P={p:.4f} | R={rec:.4f} | train_attacks={int(y_tr.sum())}")

    os.makedirs("results/figures", exist_ok=True)

    # Plot F1 vs ratio
    plt.figure()
    plt.plot(ratios, f1s, marker="o")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Anomaly ratio in training (log scale)")
    plt.ylabel("F1 score (test set)")
    plt.title("XGBoost: F1 vs training anomaly ratio")
    plt.savefig("results/figures/f1_vs_imbalance.png", bbox_inches="tight")
    plt.close()

    # (Optional) save table
    out = pd.DataFrame({"anomaly_ratio": ratios, "f1": f1s, "precision": ps, "recall": rs})
    out.to_csv("results/tables/xgb_imbalance_results.csv", index=False)
    print("✅ Saved figure: results/figures/f1_vs_imbalance.png")
    print("✅ Saved table : results/tables/xgb_imbalance_results.csv")

if __name__ == "__main__":
    main()
