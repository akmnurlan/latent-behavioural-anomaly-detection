import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import os

# Load dataset (modify path)
df = pd.read_csv("data/processed/cic_ids2017.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Train-test split (keep test balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ratios = [0.1, 0.01, 0.005, 0.001]
f1_scores = []

for ratio in ratios:
    # Downsample attacks in training set
    normal_idx = np.where(y_train == 0)[0]
    attack_idx = np.where(y_train == 1)[0]

    n_attack = int(len(normal_idx) * ratio)
    attack_sample = np.random.choice(attack_idx, min(n_attack, len(attack_idx)), replace=False)

    selected_idx = np.concatenate([normal_idx, attack_sample])

    X_train_bal = X_train[selected_idx]
    y_train_bal = y_train.iloc[selected_idx]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )

    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

    print(f"Ratio {ratio}: F1={f1:.4f}")

# Plot
plt.figure()
plt.plot(ratios, f1_scores, marker="o")
plt.xscale("log")
plt.xlabel("Anomaly Ratio (log scale)")
plt.ylabel("F1 Score")
plt.title("F1 vs Anomaly Ratio (XGBoost)")
plt.gca().invert_xaxis()

os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/f1_vs_imbalance.png")
plt.close()
