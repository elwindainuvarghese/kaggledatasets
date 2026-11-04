
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

d = load_wine()
X = pd.DataFrame(d.data, columns=d.feature_names)
f = list(X.columns)
Z = StandardScaler().fit_transform(X)
C = np.cov(Z, rowvar=False)

print("WINE DATASET")
print("Original Dataset Shape:", X.shape)
print("Original 13 Features:\n", f)
print("\nFirst 5 Rows:\n", X.head())

print("\nSTEP 1: COVARIANCE MATRIX")
print("Covariance Matrix Shape:", C.shape)
print(pd.DataFrame(C, index=f, columns=f).round(4))

v, E = np.linalg.eigh(C)
i = np.argsort(v)[::-1]
v, E = v[i], E[:, i]
pcs = [f"PC{i+1}" for i in range(13)]
exp = v/v.sum()*100
cum = np.cumsum(exp)

print("\nSTEP 2: EIGENVALUES - ALL 13 PCs")
for i in range(13):
    print(f"PC{i+1} Eigenvalue = {v[i]:.6f}")

print("\nSTEP 3: EIGENVECTORS - ALL 13 PCs")
print(pd.DataFrame(E, index=f, columns=pcs).round(4))

print("\nEXPLAINED VARIANCE - ALL 13 PCs")
print("PC   Eigenvalue   Explained %   Cumulative %")
for i in range(13):
    print(f"{pcs[i]:<5} {v[i]:.5f}       {exp[i]:.2f}          {cum[i]:.2f}")

plt.figure(figsize=(9,5))
plt.plot(range(1,14), exp, marker="o")
plt.title("Scree Plot - Wine Dataset")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.xticks(range(1,14))
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
plt.plot(range(1,14), cum, marker="o")
plt.title("Cumulative Explained Variance - Wine Dataset")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.xticks(range(1,14))
plt.grid(True)
plt.tight_layout()
plt.show()

n = 8
selected = pcs[:n]
T = E[:, :n]
ret = exp[:n].sum()
disc = 100-ret
Y = pd.DataFrame(Z @ T, columns=selected)

print("\nSTEP 4: PRINCIPAL COMPONENT SELECTION")
print("Total Original Features:", X.shape[1])
print("Total Principal Components:", len(v))
print("Selected Components:", n)
print("Selected Principal Components:", ", ".join(selected))
print(f"Variance Retained: {ret:.2f}%")
print(f"Variance Not Retained: {disc:.2f}%")
print("Discarded Components:", len(v)-n)

print("\nTRANSFORMATION MATRIX")
print(pd.DataFrame(T, index=f, columns=selected).round(4))
print("Transformation Matrix Shape:", T.shape)

print("\nSTEP 5: TRANSFORMED DATA")
print("Original Data Shape:", X.shape)
print("Reduced Data Shape:", Y.shape)
print(Y.head(10).round(4).to_string(index=False))

plt.figure(figsize=(8,6))
for t in np.unique(d.target):
    plt.scatter(Y.loc[d.target==t,"PC1"], Y.loc[d.target==t,"PC2"], label=f"Wine Class {t}", alpha=0.7, edgecolors="black")
plt.title("PCA Scatter Plot - Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
plt.bar(selected, exp[:8])
plt.title("Explained Variance of Selected 8 PCs")
plt.xlabel("Selected Principal Components")
plt.ylabel("Explained Variance (%)")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
for pc in selected:
    plt.hist(Y[pc], bins=20, alpha=0.25, density=True, edgecolor="black", label=pc)
plt.title("Distribution of Selected PC1 to PC8")
plt.xlabel("Principal Component Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

C = np.cov(Y, rowvar=False)
print("\nSTEP 6: COVARIANCE MATRIX AFTER TRANSFORMATION")
print(pd.DataFrame(C, index=selected, columns=selected).round(4))

print("\nFINAL PCA SUMMARY")
print("Original Number of Features:", X.shape[1])
print("Reduced Number of Features:", n)
print("Selected Components:", ", ".join(selected))
print(f"Variance Retained: {ret:.2f}%")
print(f"Variance Not Retained: {disc:.2f}%")
print("Original Shape:", X.shape)
print("Reduced Shape:", Y.shape)

final = pd.DataFrame({"Sample": np.arange(1,len(X)+1)})
for pc in selected:
    final[pc] = Y[pc]

print("\nFINAL REDUCED DATASET")
print(final.head(10).round(4).to_string(index=False))
final.to_csv("Wine_PCA_Reduced_8_Features.csv", index=False)
print("\nFILE SAVED")
print("Wine_PCA_Reduced_8_Features.csv")
print("\nPROGRAM COMPLETED")
