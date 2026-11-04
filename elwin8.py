
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

np.random.seed(25)
X = np.vstack([np.random.randn(50,2)*0.4+[1,7],np.random.randn(50,2)*0.6+[7,7],np.random.randn(50,2)*0.5+[4,1]])

M = minimum_spanning_tree(cdist(X,X)).toarray()
M += M.T

edges = sorted([(M[i,j],i,j) for i,j in zip(*np.where(np.triu(M)>0))],reverse=True)
T = M.copy()

for _,i,j in edges[:2]:
    T[i,j] = T[j,i] = 0

n,L = connected_components(T,directed=False)

print("MST CLUSTERING - CUSTOM DATASET 2")
print("--------------------------------")
print("Total data points:",len(X))
print("Desired number of clusters:",3)
print("Number of clusters formed:",n)

for i in range(n):
    print("Cluster",i+1,"contains",np.sum(L==i),"data points")

plt.figure(figsize=(8,5))
for i,j in zip(*np.where(np.triu(T)>0)):
    plt.plot(X[[i,j],0],X[[i,j],1],"k-",linewidth=0.7)
plt.scatter(X[:,0],X[:,1],s=25)
plt.title("Minimum Spanning Tree - Dataset 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
for i in range(n):
    plt.scatter(X[L==i,0],X[L==i,1],label="Cluster "+str(i+1),s=35)
plt.title("MST Clustering - Dataset 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
