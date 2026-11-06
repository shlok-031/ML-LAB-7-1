"""Lab 7: Naive Bayes Classification on the Iris dataset (Gaussian Naive Bayes)."""
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
# 1. Load dataset
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print("Shape:", df.shape)
print(df.head())
print("\nClass distribution:\n", df["species"].value_counts())

# 2. Split features / target and train-test split (70:30)
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
print("\nTrain size:", X_train.shape[0], " Test size:", X_test.shape[0])

# 3. Train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)
print("\nClass priors:", np.round(model.class_prior_, 3))

# 4. Predict and evaluate
y_pred = model.predict(X_test)
print("\nAccuracy: %.4f" % accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Predict a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]
print("Sample:", sample[0])
print("Predicted class:", iris.target_names[model.predict(sample)[0]])
print("Class probabilities:", np.round(model.predict_proba(sample)[0], 4))

# 6. Visualisations (light theme)
plt.style.use("default")
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=iris.target_names, cmap="Blues")
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

plt.figure(figsize=(6, 4.5))
for i, name in enumerate(iris.target_names):
    m = df["target"] == i
    plt.scatter(df.loc[m, "petal length (cm)"], df.loc[m, "petal width (cm)"], label=name)
plt.xlabel("Petal length (cm)"); plt.ylabel("Petal width (cm)")
plt.title("Iris dataset - petal length vs width"); plt.legend()
plt.savefig("scatter.png", dpi=150, bbox_inches="tight")
