
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import *

d = load_breast_cancer()
X = d.data[:, :2]
y = np.where(d.target == 0, -1, 1)

print("="*65)
print("EXPERIMENT 7: SUPPORT VECTOR MACHINE")
print("="*65)
print("Dataset: Breast Cancer Wisconsin")
print("Features: Mean Radius and Mean Texture")
print("Total samples:", len(X))

a, b, c, e = train_test_split(X, y, train_size=200, random_state=42, stratify=y)
s = StandardScaler()
a, b = s.fit_transform(a), s.transform(b)

print("\nTotal samples:", len(X))
print("Training samples:", len(a))
print("Testing samples:", len(b))

m = SVC(kernel="linear", C=1.0).fit(a, c)
w, bias = m.coef_[0], m.intercept_[0]
sv, idx = m.support_vectors_, m.support_
labels = c[idx]
w1, w2 = w
norm = np.linalg.norm(w)

print("\nLINEAR SVM: 16 CALCULATION STEPS")
print("\nSTEP 1: EQUATION OF HYPERPLANE")
print("w1*x1 + w2*x2 + b = 0")
print("\nSTEP 2: SVM CONSTRAINT")
print("y * (w1*x1 + w2*x2 + b) >= 1")

selected = np.r_[np.where(labels==1)[0][:2], np.where(labels==-1)[0][:2]]

for step, j in enumerate(selected, 3):
    p, lab = sv[j], labels[j]
    val = lab*(np.dot(w,p)+bias)
    print(f"\nSTEP {step}: SUPPORT VECTOR CONSTRAINT")
    print("Coordinates:", np.round(p,4))
    print("Class label:", lab)
    print(f"{lab} * ({w1:.4f}*({p[0]:.4f}) + {w2:.4f}*({p[1]:.4f}) + {bias:.4f}) = {val:.4f}")

print("\nSTEP 7: BIAS")
print(f"b = {bias:.6f}")
print("\nSTEP 8: WEIGHT W1")
print(f"w1 = {w1:.6f}")
print("\nSTEP 9: WEIGHT W2")
print(f"w2 = {w2:.6f}")
print("\nSTEP 10: WEIGHT VECTOR")
print("w =", np.round(w,6))
print("\nSTEP 11: OPTIMAL HYPERPLANE")
print(f"{w1:.4f}*x1 + {w2:.4f}*x2 + {bias:.4f} = 0")
print("\nSTEP 12: POSITIVE MARGIN")
print(f"{w1:.4f}*x1 + {w2:.4f}*x2 + {bias:.4f} = +1")
print("\nSTEP 13: NEGATIVE MARGIN")
print(f"{w1:.4f}*x1 + {w2:.4f}*x2 + {bias:.4f} = -1")
print("\nSTEP 14: SUPPORT VECTORS")
print("Total Support Vectors:", len(sv))
print("Support Vector Indices:", idx)
print("Coordinates:")
print(np.round(sv,4))
print("\nSTEP 15: NORM OF WEIGHT VECTOR")
print(f"||w|| = {norm:.6f}")
print("\nSTEP 16: TOTAL MARGIN WIDTH")
print(f"Margin = 2/||w|| = {2/norm:.6f}")

p = m.predict(b)
print("\nLINEAR SVM")
print("Support vectors:", len(sv))
print("Test accuracy:", round(accuracy_score(e,p)*100,2), "%")
print("\nClassification Report:")
print(classification_report(e,p,labels=[-1,1],target_names=["Malignant","Benign"],zero_division=0))
