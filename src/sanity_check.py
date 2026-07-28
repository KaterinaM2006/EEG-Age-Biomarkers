import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("results/labeled_dataset.csv")

# Collapse to binary: young vs old
df["binary_label"] = df["label"].apply(lambda x: "young" if x == "20-35" else "old")
print(df["binary_label"].value_counts())

X = df.drop(columns=["subject_id", "label", "binary_label"]).dropna(axis=1)
y = df["binary_label"]

model = RandomForestClassifier(random_state=42, class_weight="balanced")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print(f"Cross-validated accuracy per fold: {scores}")
print(f"Mean accuracy: {scores.mean():.2f}")

majority_baseline = y.value_counts(normalize=True).max()
print(f"Majority-class baseline (always guessing the bigger group): {majority_baseline:.2f}")