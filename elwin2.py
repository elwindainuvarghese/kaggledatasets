
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import *

df = pd.read_csv("CART.csv")
print(df.head())

for c in ["Age_Group", "Credit_History", "Loan_Status"]:
    df[c] = LabelEncoder().fit_transform(df[c])

X, y = df[["Age_Group", "Credit_History"]], df["Loan_Status"]
a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=42)

m = DecisionTreeClassifier(criterion="gini", random_state=42).fit(a, c)
p = m.predict(b)

e = 1 - accuracy_score(d, p)
print("\nAccuracy:", accuracy_score(d, p))
print("Error:", e)
print("Error Percentage:", e*100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(d, p))
print("\nClassification Report:")
print(classification_report(d, p))

plt.figure(figsize=(12, 7))
plot_tree(m, feature_names=X.columns, class_names=["Rejected", "Approved"], filled=True)
plt.show()
