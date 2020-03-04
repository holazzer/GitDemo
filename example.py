import numpy as np
import sklearn.datasets as ds
import sklearn.neural_network as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X,y = ds.make_circles(n_samples=1000,noise=0.05)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
