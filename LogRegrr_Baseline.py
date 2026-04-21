
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

print("=== Baseline: Logistic Regression (RAW FEATURES) ===")

# =========================
# LOAD DATA
# =========================
features_df = pd.read_parquet("processed/machine_features.parquet")
labels_df = pd.read_parquet("processed/failure_labels.parquet")

print("Features shape:", features_df.shape)
print("Labels shape:", labels_df.shape)

# =========================
# BUILD MACHINE SET
# =========================

all_machines = features_df["machine_id"].unique()
failed_machines = labels_df["machine_id"].unique()

print("\nTotal machines:", len(all_machines))
print("Failed machines:", len(failed_machines))

# =========================
# CREATE BINARY LABELS
# =========================

label_map = pd.DataFrame({
    "machine_id": all_machines
})

label_map["target"] = label_map["machine_id"].isin(failed_machines).astype(int)

print("\nFinal label distribution:")
print(label_map["target"].value_counts())

# =========================
# PREP FEATURES
# =========================

features_unique = features_df.drop_duplicates(subset=["machine_id"])

print("\nFeatures (unique machines):", features_unique.shape)

# =========================
# MERGE
# =========================

df = features_unique.merge(label_map, on="machine_id")

print("Merged shape:", df.shape)

# =========================
# FINAL DATASET
# =========================

X = df.drop(columns=["machine_id", "target"]).values
y = df["target"].values

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# TRAIN / TEST SPLIT  
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================
# TRAIN MODEL
# =========================

lr = LogisticRegression(
    max_iter=500,
    class_weight="balanced"
)

print("\nTraining Logistic Regression...")
lr.fit(X_train, y_train)

preds = lr.predict(X_test)

# =========================
# METRICS
# =========================

p = precision_score(y_test, preds, zero_division=0)
r = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)

print("\n=== Results (Test Set) ===")
print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1:        {f1:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, preds, zero_division=0))

# =========================
# INTERPRETATION
# =========================

print("\n=== Interpretation ===")
print("""
1. This baseline uses only raw machine-level features.
2. It does not incorporate graph structure (no dependencies).
3. It does not model temporal dynamics.
4. Each machine is treated independently.

This baseline predicts whether a machine will ever fail,
without modeling when or how failures propagate.

Performance differences vs ST-GNN highlight the importance of:
- Temporal modeling (cascade prediction)
- Graph structure (inter-machine dependencies)
""")

