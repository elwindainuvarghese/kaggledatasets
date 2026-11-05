
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import *

d = load_breast_cancer()
X, y = d.data, d.target

df = pd.DataFrame(X, columns=d.feature_names)
df["target"] = y
print(df.head())

a, b, c, e = train_test_split(X, y, test_size=0.2, random_state=42)

m = DecisionTreeClassifier(criterion="gini", random_state=42)
m.fit(a, c)
p = m.predict(b)

acc = accuracy_score(e, p)
err = 1 - acc

print("\nAccuracy:", acc)
print("Error:", err)
print("Error Percentage:", err*100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(e, p))

print("\nClassification Report:")
print(classification_report(e, p, target_names=d.target_names))

plt.figure(figsize=(12, 7))
plot_tree(m, feature_names=d.feature_names, class_names=d.target_names, filled=True)
plt.show()
