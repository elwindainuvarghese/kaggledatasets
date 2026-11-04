
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.tree import plot_tree

d = load_breast_cancer()
X, y = d.data, d.target
f, n = d.feature_names, d.target_names

print("="*50)
print("       RANDOM FOREST - BAGGING")
print("="*50)
print("\nDataset Name:\nBreast Cancer Wisconsin Diagnostic Dataset")
print("\nNumber of Samples:", len(X))
print("Number of Features:", X.shape[1])
print("\nClasses:\n0 =", n[0], "\n1 =", n[1])

a, b, c, e = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTraining Samples:", len(a))
print("Testing Samples :", len(b))

m = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=5, random_state=42).fit(a, c)
p = m.predict(b)
cm = confusion_matrix(e, p)
acc = accuracy_score(e, p)

print("\nRandom Forest model trained successfully!")
print("\n" + "="*30 + "\n       MODEL PERFORMANCE\n" + "="*30)
print("Algorithm       : Random Forest")
print("Number of Trees : 100")
print("Training Data   : 80%")
print("Testing Data    : 20%")
print("\nAccuracy:", round(acc*100, 2), "%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:")
print(classification_report(e, p, target_names=n))

plt.figure(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=n).plot(cmap="Blues", values_format="d")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

i = np.argsort(m.feature_importances_)[-10:]
plt.figure(figsize=(10, 7))
plt.barh(range(10), m.feature_importances_[i])
plt.yticks(range(10), f[i])
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

plt.figure(figsize=(24, 12))
plot_tree(m.estimators_[0], feature_names=f, class_names=n, filled=True, rounded=True, max_depth=3, fontsize=8)
plt.title("Decision Tree from Random Forest")
plt.tight_layout()
plt.show()

print("Dataset : Breast Cancer Wisconsin")
print("Algorithm: Random Forest (Bagging)")
print("Trees : 100")
print("Accuracy :", round(acc*100, 2), "%")
print("\nExperiment completed successfully!")
