from NeuralNet import NeuralNet
import numpy as np
import copy as cp
import time
import random

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import check_random_state


train_samples = 50000
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Pre processamento entrada
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()

# print(type(y_train))

# Arquitetura da rede
# net = NeuralNet([2, 10, 1])
# net = NeuralNet([784, 100, 100, 10]) #10300 tempo = 378
net = NeuralNet([784, 32, 32, 10]) #10300 tempo = 378
# net = NeuralNet([2, 50, 50, 50, 50, 1])  # 7650 tempo = 152
# net = NeuralNet([2, 70, 30, 30, 70, 1])  # 5310 tempo = 158
# net = NeuralNet([2, 80, 20, 20, 80, 1])  # tempo = 167


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
