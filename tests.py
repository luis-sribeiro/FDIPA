from NeuralNet import NeuralNet
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import copy as cp

#Color map for plotting figures
cmap = matplotlib.colors.ListedColormap(['blue', 'red'])

#Plotting style
matplotlib.style.use('ggplot')


#Creating the training data
n_samples = 5000
X_train , y_train = datasets.make_circles(n_samples=n_samples, random_state = 10,noise=0.15, factor=0.3)
y_train = y_train.reshape(n_samples, -1)

# X_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
# X_train = X_train.T

# y_train = np.array([[0, 1, 1, 0]])
# y_train = y_train.T

'''
plt.figure(figsize=(19.20,10.80))
plt.ylim(-1.5,1.5)
plt.xlim(-1.5,1.5)
plt.scatter(X_train[:,0], X_train[:,1], c = y_train, cmap=cmap)
plt.show()
'''
#Olhar como definir o dataSet pra entrada pra rede neural
#dataSet = [X_train, y_train]

#Acredito que essa seja a forma certa de criar o dataSet
#Dentro da funcao SGD ele transofrma essa entrada em uma lista
dataSet = zip(X_train, y_train)

dataSet1 = cp.copy(dataSet)
dataSet1 = list(dataSet1)

net = NeuralNet([2, 5, 1])
 
epochs = 1000
mini_batch_size = 100
# mini_batch_size = n_samples
eta = 0.1

print("comeco:")
#net.SGD(zip(X_train[0:400,:], y_train[0:400]), epochs, mini_batch_size, eta)
# net.SGD2(X_train, y_train, epochs, mini_batch_size, eta)
net.trainFDIPA(X_train, y_train, epochs, mini_batch_size, eta)
print("fim")


#Teste:
'''
for i in range(400,500):
   r = net.feedforward(X_train[i,:])
   print("rede: %r \t yEsperado = %r"% (r[0], y_train[i]))
'''