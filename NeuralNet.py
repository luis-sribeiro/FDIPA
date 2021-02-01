# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 13:28:51 2020

@author: vitor
"""

import numpy as np
import random

#classe que representa uma rede neural
class NeuralNet(object):

    def __init__(self, sizes):
        """sizes contem o numero de neuronios em cada camada. 
        Ex: se sizes = [2,3,1]  entao a rede vai ter 2 neuronios
        na primeira camada, 3 neuronios na segunda camada e 
        1 neuronio na terceira camada.
        
        """
        self.num_layers = len(sizes) # numero de camadas da rede
        self.sizes     = sizes      # numero de neuronios na respectiva camada
        aux = np.ones(len(sizes), dtype = np.int8) #para adicionar 1 neuronio em cada camada exceto na ultima
        aux[-1] = 0
        self.sizes_com_bias = sizes + aux
        self.weights   = [np.random.randn(y,x) for x, y in zip(self.sizes_com_bias[:-1], self.sizes_com_bias[1:])]
    
    #faz o feedfoward em um conjunto de entrada
    def feedfowardbatch(self, a):
        for i in range(0, len(self.weights)-1):
            a = sigmoid(np.dot(self.weights[i],a))
            a[0 , :] = 1
        return sigmoid(np.dot(self.weights[-1],a))
    
    def 
    
    def feedforward(self, a):
        #força o primeiro neuronio de cada camada ser 1 uma vez que representa o bias
        a = np.concatenate(([1], a), axis=0)
        for i in range(0,len(self.weights)-1):
            a = sigmoid(np.dot(self.weights[i], a))
            a[0] = 1
        return sigmoid(np.dot(self.weights[-1],a))
        
    #Stochastic Gradient Descent
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        training_data = list(training_data) #lista (x,y) que representam 
                                            #entradas(x) e saídas desejadas(y) para treinamento.
        n = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} finalizada".format(j))
       
        
    def update_mini_batch(self, mini_batch, eta):
        """ Atualiza os pesos e bias da rede aplicando 
        a descida do gradiente usando backpropagation para um único mini lote.
        O 'mini-batch' é uma lista de tuplas '(x,y)', 'eta' é a taxa de aprendizado.        """

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_w = self.backprop(x, y)            
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

    def backpropBloco(self, mini_batch):
        """Retorna `nabla_w` representando o
         gradiente para a função de custo C_x. `nabla_w` é uma lista de camadas de matrizes numpy,
         semelhante a  `self.weights`."""

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Retirar do mini_batch o conjunto dos x's de entrada para montar  o activation com
        # a primeira linha formada por 1's
        # Feedforward
        activation = np.zeros(len(x)+1)
        activation[0] = 1
        activation[1:] = x

        # Lista para armazenar todas as ativações, camada por camada
        activations = [activation] 

        # Lista para armazenar todos os vetores z, camada por camada
        zs = [] 
        for i in range(0,len(self.weights)-1):
            w = self.weights[i]
            z = np.dot(w, activation)
            zs.append(z)
            activation = sigmoid(z)
            activation[0,:] = 1
            activations.append(activation)

        w = self.weights[-1]
        z = np.dot(w, activation)
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Aqui, l = 1 significa a última camada de neurônios, l = 2 é a
        # segunda e assim por diante. 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return nabla_w        

    def backprop(self, x, y):
        """Retorna `nabla_w` representando o
         gradiente para a função de custo C_x. `nabla_w` é uma lista de camadas de matrizes numpy,
         semelhante a  `self.weights`."""

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward
        activation = np.zeros(len(x)+1)
        activation[0] = 1
        activation[1:] = x

        # Lista para armazenar todas as ativações, camada por camada
        activations = [activation] 

        # Lista para armazenar todos os vetores z, camada por camada
        zs = [] 
        for i in range(0,len(self.weights)-1):
            w = self.weights[i]
            z = np.dot(w, activation)
            zs.append(z)
            activation = sigmoid(z)
            activation[0] = 1
            activations.append(activation)

        w = self.weights[-1]
        z = np.dot(w, activation)
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Aqui, l = 1 significa a última camada de neurônios, l = 2 é a
        # segunda e assim por diante. 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return nabla_w

    def evaluate(self, test_data):
        """Retorna o número de entradas de teste para as quais a rede neural 
         produz o resultado correto. Note que a saída da rede neural
         é considerada o índice de qualquer que seja
         neurônio na camada final que tenha a maior ativação."""

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Retorna o vetor das derivadas parciais."""
        return (output_activations-y)

    #converte matriz para vetor
    def mat2vet(self):
        return np.concatenate(list(map(lambda a: a.reshape(-1), self.weights))) #concatena lista de arrays

    #converte vetor para matriz    
    def vet2mat(self,v):
        start = 0
        weightsAux = []
        for w in self.weights:
            end = start + w.size
            weightsAux.append(v[start:end].reshape(w.shape))
            start = end
            
    #Nesse caso, deverá ser realizada a transformacao antes
    #de fazer a computacao da rede
    #To Do: obter o tamanho do data set, decidir como o x e o y serão passados para decidir como o calculo será feito

    def feedforwardFDIPA(self, x, y, w, mini_batch):
        n = 500 #Tamanho do dataset alguma forma de obter esse tamanho de dataSet
        x = #alguma forma de transformar o x em uma matriz com os dados de treinamento
        a = cp.copy(x)
        start = 0
        for i in range(0, len(self.weights)-1):
            end = start + self.weights[i].size
            a = sigmoid(np.dot(w[0,start:end].reshape(self.weights[i].shape) ,a))
            start = end
            a[0 , :] = 1
        a = sigmoid(np.dot(w[0,start:], a))
        return (1/(2*n))*np.linalg.norm(a-y,'fro'), -(np.linalg.norm(w)**2)/2
#        for x, y in mini_batch:	
#            retornoAux = np.zeros(len(x) + 1)
#            retornoAux[1:] = np.copy(x)
#            #força o primeiro neuronio de cada camada ser 1 uma vez que representa o bias
#            retornoAux[0] = 1
#            contador = 0
#            for i in range(0,len(self.weights)-1):
#                retornoAux = sigmoid(np.dot(w[0,contador:contador+self.weights[i].size].reshape(self.weights[i].shape) , retornoAux))
#                contador = contador + self.weights[i].size + 1
#                retornoAux[0] = 1
#            retornoAux = sigmoid(np.dot(w[-1], retornoAux))
#            retorno = retorno + np.linalg.norm(retornoAux - y)**2
#        return (1/(2*len(mini_batch))) * retorno, -(np.linalg.norm(w)**2)/2

# funcao de ativacao sigmoide
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
    
# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

rede1 = NeuralNet([2,3,1])