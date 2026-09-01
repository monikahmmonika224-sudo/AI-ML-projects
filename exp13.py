import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
iris=datasets.load_iris()
x=iris.data
y=iris.target
print(x.shape)
print(y.shape)
pca=PCA(n_components=2)
pca.fit(x)
print(pca.components_)
x=pca.transform(x)
print(x.shape)
plt.scatter(x[:,0],x[:,1],c=y)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
res=DecisionTreeClassifier()
res.fit(x_train,y_train)
y_predict=res.predict(x_test)
print(accuracy_score(y_test,y_predict))
plt.show()