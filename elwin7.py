
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import *

d = load_breast_cancer()
X, y = d.data[:, :2], np.where(d.target==0,-1,1)

a,b,c,e = train_test_split(X,y,train_size=200,random_state=42,stratify=y)
s = StandardScaler()
a,b = s.fit_transform(a),s.transform(b)

models = {
    "Polynomial SVM": SVC(kernel="poly",degree=3,gamma="scale",coef0=1.0,C=1.0),
    "Gaussian SVM": SVC(kernel="rbf",gamma="scale",C=1.0),
    "Sigmoid SVM": SVC(kernel="sigmoid",gamma="scale",coef0=0.0,C=1.0)
}

acc, counts = {}, {}

for name,m in models.items():
    m.fit(a,c)
    p = m.predict(b)
    acc[name] = accuracy_score(e,p)*100
    counts[name] = len(m.support_vectors_)
    print("\n"+name.upper())
    print("Support vectors:",counts[name])
    print(f"Test accuracy: {acc[name]:.2f}%")
    print("\nClassification Report:")
    print(classification_report(e,p,labels=[-1,1],target_names=["Malignant","Benign"],zero_division=0))

x,y = a.min(axis=0)-0.5,a.max(axis=0)+0.5
xx,yy = np.meshgrid(np.linspace(x[0],y[0],300),np.linspace(x[1],y[1],300))
grid = np.c_[xx.ravel(),yy.ravel()]

for name,m in models.items():
    z = m.decision_function(grid).reshape(xx.shape)
    plt.figure(figsize=(9,6))
    plt.contourf(xx,yy,z,levels=[-np.inf,0,np.inf],alpha=0.15,cmap="coolwarm")
    plt.contour(xx,yy,z,levels=[-1,0,1],colors=["black","red","black"],linestyles=["--","-","--"])
    plt.scatter(a[c==-1,0],a[c==-1,1],label="Malignant (-1)")
    plt.scatter(a[c==1,0],a[c==1,1],label="Benign (+1)")
    plt.scatter(m.support_vectors_[:,0],m.support_vectors_[:,1],s=140,facecolors="none",edgecolors="black",label="Support Vectors")
    plt.title(name+" Decision Boundary")
    plt.xlabel("Mean Radius (Standardized)")
    plt.ylabel("Mean Texture (Standardized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\nFINAL KERNEL COMPARISON")
print("Training samples:",len(a))
print("Testing samples:",len(b))
for name in models:
    print(f"{name}: Support vectors = {counts[name]}, Accuracy = {acc[name]:.2f}%")

plt.figure(figsize=(10,5))
bars = plt.bar(acc.keys(),acc.values())
plt.title("SVM Kernel Test Accuracy Comparison")
plt.ylabel("Test Accuracy (%)")
plt.xlabel("SVM Kernel")
plt.ylim(0,105)
for bar,v in zip(bars,acc.values()):
    plt.text(bar.get_x()+bar.get_width()/2,v+2,f"{v:.2f}%",ha="center")
plt.grid(axis="y",alpha=0.4)
plt.tight_layout()
plt.show()
print("\nEXPERIMENT COMPLETED")
