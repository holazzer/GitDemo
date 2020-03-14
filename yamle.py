import numpy as np
import sklearn.datasets as ds
from sklearn.neighbors import KNeighborsClassifier as K
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

X,y = ds.make_circles(n_samples=1000,noise=0.05)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y)

clf = K()

clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))

cmap = ListedColormap(["#cccccc","#eeeeee"])

a = X.min()
b = X.max()
h = 0.02
xx,yy = np.meshgrid( np.arange(a,b,h),np.arange(a,b,h) )
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
zz = Z.reshape(xx.shape)

plt.pcolormesh(xx,yy,zz,cmap=cmap)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()