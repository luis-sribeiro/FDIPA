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
#plt.show()
#Olhar como definir o dataSet pra entrada pra rede neural
#dataSet = [X_train, y_train]

#Acredito que essa seja a forma certa de criar o dataSet
#Dentro da funcao SGD ele transofrma essa entrada em uma lista
dataSet = zip(X_train, y_train)
dataSet1 = list(dataSet)

#for x,y in dataSet:
#    print("x: %r \t y = %r"% (x, y))

net = NeuralNet([2,5,1])

net.feedforward(X_train[0])
net.SGD(dataSet, 100, 25, 0.8)

print("fim")

#Teste:
for x,y in dataSet1:
    r = net.feedforward(x)
    print("x: %r \t y = %r"% (r, y))
