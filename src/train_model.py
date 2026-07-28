import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("results/labeled_dataset.csv")

print("Label distribution:")
print(df["label"].value_counts())
print()

# Separate features (X) from the label we want to predict (y)
X = df.drop(columns=["subject_id", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} subjects, testing on {len(X_test)} subjects")

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print("Class order:", model.classes_)

print("\nDetailed report:")
print(classification_report(y_test, y_pred))