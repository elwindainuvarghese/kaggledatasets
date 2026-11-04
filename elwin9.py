
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

d = load_breast_cancer()
np.random.seed(42)
X = d.data[np.random.choice(len(d.data),150,replace=False)]
Z = StandardScaler().fit_transform(X)
P = PCA(n_components=2).fit_transform(Z)
L = DBSCAN(eps=1.5,min_samples=5).fit_predict(P)

n = len(set(L))-(-1 in L)
noise = np.sum(L==-1)

print("DBSCAN CLUSTERING RESULTS")
print("Dataset: Breast Cancer Dataset")
print("Total data points:",len(X))
print("Number of original features:",X.shape[1])
print("Number of reduced dimensions:",P.shape[1])
print("Epsilon (eps): 1.5")
print("Minimum samples: 5")
print("Number of clusters formed:",n)
print("Number of noise points:",noise)

for i in sorted(set(L)):
    if i!=-1:
        print("Cluster",i+1,"contains",np.sum(L==i),"data points")

plt.figure(figsize=(8,5))
for i in sorted(set(L)):
    plt.scatter(P[L==i,0],P[L==i,1],label="Noise" if i==-1 else "Cluster "+str(i+1),marker="x" if i==-1 else "o",s=60 if i==-1 else 40)

plt.title("DBSCAN Clustering - Breast Cancer Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
