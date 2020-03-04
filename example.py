import numpy as np
import sklearn.datasets as ds
import sklearn.neural_network as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from matplotlib.colors import ListedColormap

X,y = ds.make_circles(n_samples=1000,noise=0.05)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y)

clf = nn.MLPClassifier(hidden_layer_sizes=(50,20),solver='lbfgs',random_state=1,verbose=1)

clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))


# 绘制colormap的方法参考:
# https://www.jianshu.com/p/f544ac592979

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
