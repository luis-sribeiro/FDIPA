from NeuralNet import NeuralNet
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import copy as cp
import time

import random

#Color map for plotting figures
cmap = matplotlib.colors.ListedColormap(['blue', 'red'])

#Plotting style
matplotlib.style.use('ggplot')

#Creating the training data
n_samples = 2000
X_train , y_train = datasets.make_circles(n_samples=n_samples, random_state = 10,noise=0.15, factor=0.3)
y_train = y_train.reshape(n_samples, -1)

'''
# Plot the dataset
plt.figure(figsize=(19.20,10.80))
plt.ylim(-1.5,1.5)
plt.xlim(-1.5,1.5)
plt.scatter(X_train[:,0], X_train[:,1], c = y_train, cmap=cmap)
plt.show()
'''

#Acredito que essa seja a forma certa de criar o dataSet
#Dentro da funcao SGD ele transofrma essa entrada em uma lista
dataSet = zip(X_train, y_train)

dataSet1 = cp.copy(dataSet)
dataSet1 = list(dataSet1)

# net = NeuralNet([2, 10, 1])
# net = NeuralNet([2, 100, 100, 1]) #10300 tempo = 378
# net = NeuralNet([2, 50, 50, 50, 50, 1]) # 7650 tempo = 152
# net = NeuralNet([2, 70, 30, 30, 70, 1]) #5310 tempo = 158
net = NeuralNet([2, 80, 20, 20, 80, 1])  # tempo = 167


epochs = 1000
mini_batch_size = 100

eta = 0.5
iteracoes = 2

print("Comeco:")
ini = time.time()
net.SGD(X_train, y_train, epochs, mini_batch_size, eta)
# net.trainFDIPA(X_train, y_train, epochs, mini_batch_size, eta, iteracoes)
end = time.time()
print("Fim")
print("Tempo: \t\t" + str(end-ini) + " (s)" )
