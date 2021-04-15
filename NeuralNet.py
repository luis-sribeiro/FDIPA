import numpy as np
import random
import copy as cp
import time

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
        uns = np.ones((np.shape(a)[0],1))
        a = np.concatenate((uns, a), axis = 1)
        a = a.T
        for i in range(0, len(self.weights)-1):
            a = sigmoid(np.dot(self.weights[i],a))
            a[0 , :] = 1
        
        # return sigmoid(np.dot(self.weights[-1],a))
        # output = sigmoid(np.dot(self.weights[-1],a))
        output = np.dot(self.weights[-1],a)
        output = np.exp(output)

        soma = np.sum(output, axis = 0)
        # cada linha do output eh saida de um dado do dataset
        retorno = output/ soma[None, :]
        return retorno

    def feedforward(self, a):
        #força o primeiro neuronio de cada camada ser 1 uma vez que representa o bias
        a = np.concatenate(([1], a), axis=0)
        a = a.T
        for i in range(0,len(self.weights)-1):
            a = sigmoid(np.dot(self.weights[i], a))
            a[0] = 1
        return sigmoid(np.dot(self.weights[-1], a))

    def trainFDIPA(self, X, y, epochs, mini_batch_size, eta, iteracoes, X_test, y_test):
        n = len(X)
        display_step = 50

        for j in range(epochs):
            p = np.random.permutation(len(X))
            X = X[p]
            y = y[p]

            # cont = 0
            for k in range(0, n, mini_batch_size):
                self.update_mini_batch_FDIPA(X[k:k+mini_batch_size], y[k:k+mini_batch_size], eta, iteracoes)
                # cont += 1
                # print(cont)

            if j % display_step == 0:
                # predictions = np.asarray([self.feedforward(x) for x in X])
                predictions_train = (self.feedfowardbatch(X)).T
                predictions_test = (self.feedfowardbatch(X_test)).T
                accuracy_train = np.sum(((predictions_train >= 0.5) == y))/X.shape[0]
                accuracy_test = np.sum(((predictions_test >= 0.5) == y_test))/X_test.shape[0]                
                print("Epoch {}/{}  | Acurácia de Treino {:.4f} | Acurácia de Teste {:.4f}".format(j+1, epochs, accuracy_train, accuracy_test))

    def update_mini_batch_FDIPA(self, x_train, y_train, eta, iteracoes):
        L0 = np.ones([1])
        # ini = time.time()
        self.weights = self.FDIPA(self.weights, L0, iteracoes, x_train, y_train, eta)
        # end = time.time()
        # print("Tempo  FDIPA: \t\t" + str(end-ini) + " (s)" )

    #Stochastic Gradient Descent
    def SGD(self, X, y, epochs, mini_batch_size, eta, test_data = None):
        n = len(X)
        display_step = 1
        for j in range(epochs):
            p = np.random.permutation(len(X))
            X = X[p]
            y = y[p]

            for k in range(0, n, mini_batch_size):
                self.update_mini_batch(X[k:k+mini_batch_size], y[k:k+mini_batch_size], eta)

            if j % display_step == 0:
                predictions = (self.feedfowardbatch(X)).T
                # predictions = np.asarray([self.feedforward(x) for x in X])
                accuracy = np.sum(np.argmax(predictions, axis = 1) == np.argmax(y, axis = 1))/X.shape[0]
                # accuracy = np.sum(((predictions >= 0.5) == y))/X.shape[0]
                print("Epoch {}/{}  | Acurácia {:.4f}".format(j, epochs, accuracy))
          
    def update_mini_batch(self, x_train, y_train, eta):
        """ Atualiza os pesos e bias da rede aplicando 
        a descida do gradiente usando backpropagation para um único mini lote.
        'eta' é a taxa de aprendizado.        """

        nabla_w = self.backpropFDIPA(self.weights, x_train, y_train)
        self.weights = [w-(eta/len(x_train))*nw for w, nw in zip(self.weights, nabla_w)]

    # realiza o backpropagation ponto a ponto ( para um x e um y)
    def backprop(self, x, y):
        """Retorna `nabla_w` representando o
         gradiente para a função de custo C_x. `nabla_w` é uma lista de camadas de matrizes numpy,
         semelhante a  `self.weights`."""

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward
        activation = np.zeros((1,len(x)+1))
        activation[0,0] = 1
        activation[0,1:] = x
        activation = activation.T
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
        # penultima e assim por diante. 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())            
        return nabla_w

    # realiza o backpropagation para um conjunto de dados x e y
    def backpropFDIPA(self, w, x, y):
        """Retorna `nabla_w` representando o
         gradiente para a função de custo C_x. `nabla_w` é uma lista de camadas de matrizes numpy,
         semelhante a  `self.weights`."""
        nabla_w = [np.zeros(waux.shape) for waux in w]
        
        # Feedforward
        activation = x
        uns = np.ones((np.shape(activation)[0],1))
        activation = np.concatenate((uns, activation), axis = 1)
        activation = activation.T
        # Lista para armazenar todas as ativações, camada por camada
        activations = [activation] 
        y = y.T

        # Lista para armazenar todos os vetores z, camada por camada
        zs = [] 
        for i in range(0,len(w)-1):
            z = np.dot(w[i], activation)
            zs.append(z)
            activation = sigmoid(z)
            activation[0,:] = 1
            activations.append(activation)

        z = np.dot(w[-1], activation)
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
            delta = np.dot(w[-l+1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return nabla_w        

    def evaluate(self, test_data):
        """Retorna o número de entradas de teste para as quais a rede neural 
         produz o resultado correto. Note que a saída da rede neural
         é considerada o índice de qualquer que seja
         neurônio na camada final que tenha a maior ativação."""

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #output_activations representa a computação da rede e y os dados de treinamento
    def cost(self, output_activations, y):
        n = len(y)
        # return (1/(2*n))*(np.linalg.norm(output_activations - y.T))**2
        #se y tem dim nx1
        index = np.argmax(y, axis = 1)
        output = output_activations[np.arange(len(index)), index]
        return (1/n)*np.sum(output)

    def cost_derivative(self, output_activations, y):
        """Retorna o vetor das derivadas parciais."""
        n = len(y)
        # return (output_activations-y)
        index = np.argmax(y, axis = 1)
        output = output_activations[np.arange(len(index)), index]
        return (1/n)* np.sum(1/output)

    #converte matriz para vetor
    def mat2vet(self):
        retorno = np.concatenate(list(map(lambda a: a.reshape(-1), self.weights))) #concatena lista de arrays
        retorno = retorno.reshape(retorno.size, 1)
        return retorno

    #converte vetor para matriz    
    def vet2mat(self,v):
        start = 0
        weightsAux = []
        for w in self.weights:
            end = start + w.size
            weightsAux.append(v[start:end].reshape(w.shape))
            start = end
        return weightsAux
            
    #Nesse caso, deverá ser realizada a transformacao antes
    #de fazer a computacao da rede
    #Função que queremos minimizar
    def feedforwardFDIPA(self, w, x, y):
        """Retorna o f do FDIPA"""

        uns = np.ones((np.shape(x)[0],1))
        a = np.concatenate((uns, x), axis = 1)
        a = a.T

        for i in range(0, len(w)-1):
            a = sigmoid(np.dot(w[i] ,a))
            a[0 , :] = 1
        a = sigmoid(np.dot(w[-1], a))
        return self.cost(a, y)

    #w nesse caso pode ser o vetor
    def g_FDIPA(self, w):
        """Retorna a restrição g do FDIPA
        Recebe w como um vetor"""
        # r = 5
        # g = np.linalg.norm(mat2vet(w)) - r
        g = -(np.linalg.norm(w)**2)/2
        return g

    #retorna df e dg pro FDIPA(dfun)
    def df_FDIPA(self, w, x, y):
        """Retorna a derivada da função f. 
        Recebe w como uma lista de matrizes"""
        return mat2vet(self.backpropFDIPA(w, x, y))

    def dg_FDIPA(self, w):
        """Retorna derivada da restrição g. 
        Recebe w como um vetor"""
        return -w

    def ddfunFDIPA(self, w, L):
        return np.eye(len(w))
    
    #w0 passado eh o w inicial
    #w0 eh passado na forma de matrizes do mesmo formato dos pesos da rede
    #x e y sao dados de treinamento
    def  FDIPA(self, w0, L0, tol, x_train, y_train, eta):
        #Dados iniciais

        #x0 converte w0 na forma de vetor
        x0 = mat2vet(w0)
        
        f0 = self.feedforwardFDIPA(w0, x_train, y_train)
        g0 = self.g_FDIPA(x0)

        df0 = self.df_FDIPA(w0, x_train, y_train)
        dg0 = self.dg_FDIPA(x0)

        # B0 = self.ddfunFDIPA(x0, L0)

        # n = len(x0)
        m = len(L0)

        # Caso queira mais de uma restrição
        E = np.ones([m,1])        

        phi = .1 #Multiplica a norma de d1
        epsi = 0.8

        # d1 = np.ones_like(x0)

        # Inicio do Programa FDIPA        
        cont = 0
        while cont < tol :
            cont = cont + 1
            # Calculo da direção
                         
            norm_dg0 = np.linalg.norm(dg0)**2
            div = (g0/L0 - norm_dg0)

            dx1 = -(df0 + (dg0.dot(dg0.T.dot(df0) )) / div)
            # dx1 = -df0 - (dg0.T.dot(df0)) * dg0 / div
            
            if np.linalg.norm(dx1) < 10**(-6):
                # print("saida1")
                return self.vet2mat(x0)
            else:
                #Segunda direcao d_beta
                dx2 = (L0/g0)*(dg0 + (dg0.dot(norm_dg0))/div)

                if  (df0.T).dot(dx2) > 0:
                    #rho
                    r0 = min( [phi*np.linalg.norm(dx1)**2, ((epsi-1)*(df0.T).dot(dx1))/(df0.T.dot(dx2))])
                else:
                    r0 = phi*(np.linalg.norm(dx1)**2)               

                # Direcao de busca
                dx = dx1 + r0*dx2
                # d  = dx
                
                t  = 1 
                # t  = 0.01

                xn = x0 + t*dx

                Lx1 = -(L0/g0)*(dg0.T.dot(dx1))
                Lx2 = -(L0/g0)*(dg0.T.dot(dx2)+E)
                L   = np.abs(Lx1+r0*Lx2)

                mat = self.vet2mat(xn)
                fn = self.feedforwardFDIPA(mat, x_train, y_train)

                # ini = time.time()
                while ((fn-f0) > 0):
                    t = 0.7 * t
                    xn = x0 + t * dx
                    mat = self.vet2mat(xn)
                    fn = self.feedforwardFDIPA(mat, x_train, y_train)

                print("t = " + str(t))
                # end = time.time()
                # print("Tempo Busca Linear: \t\t" + str(end-ini) + " (s)" )

                # Criterio de Parada 
                # Parada Forcada   
                if (np.linalg.norm(f0-fn) < 10**(-6)):
                    # print("saida2")
                    return self.vet2mat(xn)
                x0  = xn
                
                f0 = fn
                g0 = self.g_FDIPA(x0)

                mat = self.vet2mat(x0)
                df0 = self.df_FDIPA(mat, x_train, y_train)
                dg0 = self.dg_FDIPA(x0)

                L0 = L + 10**(-8)
        return self.vet2mat(xn)
        # return [xn, L, fn, gn, counter, t, d, r0]

def f_activation(z):
    pass

def df_activation(z):
    pass

# funcao de ativacao sigmoide
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
    
# Função para retornar as derivadas da função Sigmoide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#converte matriz para vetor
def mat2vet(w):
    retorno = np.concatenate(list(map(lambda a: a.reshape(-1), w))) #concatena lista de arrays
    retorno = retorno.reshape(retorno.size, 1)
    return retorno