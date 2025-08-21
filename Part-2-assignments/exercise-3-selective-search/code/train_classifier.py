import json
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import os

Feature_File = "results/features.json"
Model_Path = "results/svm_model.joblib"

# Data Loading
with open(Feature_File, "r") as f:
    data = json.load(f)

X = np.array([d["feature"] for d in data])
y = np.array([d["label"] for d in data])

# Data Splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# SVM Evaluate
y_pred = clf.predict(X_val)
print("SVM Evaluation Report:")
print(classification_report(y_val, y_pred))

# Save model
os.makedirs(os.path.dirname(Model_Path), exist_ok=True)
joblib.dump(clf, Model_Path)
print(f"SVM model Saved to: {Model_Path}")
