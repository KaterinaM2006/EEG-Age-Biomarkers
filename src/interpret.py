import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("results/labeled_dataset.csv")

X = df.drop(columns=["subject_id", "label"]).dropna(axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# TreeExplainer is a fast, exact SHAP method built specifically for tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"Class order: {model.classes_}")
print(f"shap_values type: {type(shap_values)}, shape: {np.array(shap_values).shape}")

for i, class_name in enumerate(model.classes_):
    if isinstance(shap_values, list):
        class_shap_values = shap_values[i]
    else:
        class_shap_values = shap_values[:, :, i]

    plt.figure()
    shap.summary_plot(class_shap_values, X_test, show=False, max_display=15)
    plt.title(f"SHAP summary - predicting '{class_name}'")
    plt.tight_layout()
    plt.savefig(f"results/figures/shap_summary_{class_name}.png", dpi=150)
    plt.close()
    print(f"Saved results/figures/shap_summary_{class_name}.png")