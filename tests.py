from NeuralNet import NeuralNet
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


#Color map for plotting figures
cmap = matplotlib.colors.ListedColormap(['blue', 'red'])
x = 6
#Plotting style
matplotlib.style.use('ggplot')


#Creating the training data
X_train , y_train = datasets.make_circles(n_samples=500, random_state = 10,noise=0.15, factor=0.3)

plt.figure(figsize=(19.20,10.80))
plt.ylim(-1.5,1.5)
plt.xlim(-1.5,1.5)
plt.scatter(X_train[:,0], X_train[:,1], c = y_train, cmap=cmap)
plt.show()

#Add coluna de uns na entrada de treinamento
uns = np.ones((np.shape(X_train)[0],1))
X_train_aux = np.concatenate((uns, X_train), axis = 1)

net = NeuralNet([2,5,1])

net.feedforward(X_train[0])