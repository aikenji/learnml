import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# read data from iris.data
df = pd.read_csv('iris.data', header=None)
# print(df.tail())

# select setosa and versicolor
Y = df.iloc[0:100, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1], 
			color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], 
			color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# learning using perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, Y)

plt.plot(range(1, len(ppn.errors_)+1), 
		ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('number of updates')
plt.show()




