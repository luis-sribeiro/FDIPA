import numpy as np
import random
import copy as cp

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
    
    def feedforward(self, a):
        #força o primeiro neuronio de cada camada ser 1 uma vez que representa o bias
        a = np.concatenate(([1], a), axis=0)
        a = a.T
        for i in range(0,len(self.weights)-1):
            a = sigmoid(np.dot(self.weights[i], a))
            a[0] = 1
        return sigmoid(np.dot(self.weights[-1], a))

    def trainFDIPA(self, X, y, epochs, mini_batch_size, eta, test_data = None):
        n = len(X)
        display_step = 50
        for j in range(epochs):
            p = np.random.permutation(len(X))
            X = X[p]
            y = y[p]
            mini_batches = [(list(zip(X[k:k+mini_batch_size], y[k:k+mini_batch_size]))) for k in range(0, n, mini_batch_size)]
            # predictions = np.asarray([self.feedforward(x) for x in X])
            # accuracy = np.sum(((predictions >= 0.5) == y))/X.shape[0]
            # print("Inicio | Acurácia {:.4f}".format(accuracy))
            # cont = 0
            for mini_batch in mini_batches:
                # cont += 1            
                self.update_mini_batch_FDIPA(mini_batch, eta)
                # print(cont)

            if j % display_step == 0:
                predictions = np.asarray([self.feedforward(x) for x in X])
                accuracy = np.sum(((predictions >= 0.5) == y))/X.shape[0]
                print("Epoch {}/{}  | Acurácia {:.4f}".format(j, epochs, accuracy))
        # training_data = list(training_data) #lista (x,y) que representam 
        #                                     #entradas(x) e saídas desejadas(y) para treinamento.
        # n = len(training_data)
        # if test_data:
        #     test_data = list(test_data)
        #     n_test = len(test_data)

        # for j in range(epochs):
        #     random.shuffle(training_data)
        #     mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                        
        #     for mini_batch in mini_batches:            
        #         self.update_mini_batch(mini_batch, eta)
            
        #     if test_data:
        #        print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
        #     else:
        #        print("Epoch {} finalizada".format(j))

    def update_mini_batch_FDIPA(self, mini_batch, eta):
        """ Atualiza os pesos e bias da rede aplicando 
        a descida do gradiente usando backpropagation para um único mini lote.
        O 'mini-batch' é uma lista de tuplas '(x,y)', 'eta' é a taxa de aprendizado.        """

        # nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # for x, y in mini_batch:
        #     delta_nabla_w = self.backprop(x, y)            
        #     nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #Obtem matriz dos dados de entrada separado X e Y         
        x_train = np.zeros((len(mini_batch), len(mini_batch[0][0])))
        y_train = np.zeros((len(mini_batch), len(mini_batch[0][1])))
        
        for i in range(0, len(mini_batch)):
            x_train[i] = mini_batch[i][0]
            y_train[i] = mini_batch[i][1]
        
        L0 = np.ones([1])
        tol = 5
        self.weights = self.FDIPA(self.weights, L0, tol, x_train, y_train)

    #To Do: Trocar o parametro "training_data" por x_train e y_train separado 
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
            
            #for i in range(0,)
            #    self.update_mini_batch(x_train[k+tam], y_train[k+tam], eta)
            
            for mini_batch in mini_batches:            
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
               print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
               print("Epoch {} finalizada".format(j))

            #if j % 100 == 0:
            #    print("Epoch {}/{}  | Acurácia {}".format(j, epochs, self.feedfowardbatch(training_data)))
          
    #Todo : Atualizar essa funcao para trocar  o parametro "mini_batch" por "x_train" e "y_train"
    def update_mini_batch(self, mini_batch, eta):
        """ Atualiza os pesos e bias da rede aplicando 
        a descida do gradiente usando backpropagation para um único mini lote.
        O 'mini-batch' é uma lista de tuplas '(x,y)', 'eta' é a taxa de aprendizado.        """

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # for x, y in mini_batch:
        #     delta_nabla_w = self.backprop(x, y)            
        #     nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #Obtem matriz dos dados de entrada separado X e Y         
        x_train = np.zeros((len(mini_batch), len(mini_batch[0][0])))
        y_train = np.zeros((len(mini_batch), len(mini_batch[0][1])))
        
        for i in range(0, len(mini_batch)):
            x_train[i] = mini_batch[i][0]
            y_train[i] = mini_batch[i][1]

        nabla_w = self.backpropFDIPA(self.weights, x_train, y_train)

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

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

    def cost_derivative(self, output_activations, y):
        """Retorna o vetor das derivadas parciais."""
        return (output_activations-y)

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
    #To Do: obter o tamanho do data set, decidir como o x e o y serão passados para decidir como o calculo será feito
    #retorna f e g pro FDIPA (fun)
    #Retorna a funcao do erro quadratico e -(norma de w) ^2 / 2
    def feedforwardFDIPA(self, w, x, y):
        n = len(x) #Tamanho do mini_batch

        a = cp.copy(x)
        uns = np.ones((np.shape(a)[0],1))
        a = np.concatenate((uns, a), axis = 1)
        a = a.T
        

        for i in range(0, len(w)-1):
            a = sigmoid(np.dot(w[i] ,a))
            a[0 , :] = 1
        a = sigmoid(np.dot(w[-1], a))
        
        # ret = a-y.T
        # ret = np.linalg.norm(ret)        
        return (1/(2*n))*(np.linalg.norm(a - y.T))**2, -(np.linalg.norm(mat2vet(w))**2)/2

    #retorna df e dg pro FDIPA(dfun)
    def dfunFDIPA(self, w, x, y):
        #recebe w como uma lista de matrizes
        return mat2vet(self.backpropFDIPA(w, x, y)), -mat2vet(w)
    
    def ddfunFDIPA(self, w, L):
        return np.eye(len(w))
    
    #w0 passado eh o w inicial
    #w0 eh passado na forma de matrizes do mesmo formato dos pesos da rede
    #x e y sao dados de treinamento
    def  FDIPA(self, w0, L0, tol, x_train, y_train):
        ################################################################
        ###################      Dados iniciais     ####################
        ################################################################
        # print("INICIO FDIPA")
        counter = np.zeros([5])
        Buscatol = 0
        vc = 0

        #x0 converte w0 na forma de vetor
        x0 = mat2vet(w0)
        f0 , g0 = self.feedforwardFDIPA(w0, x_train, y_train)

        counter[1] = counter[1] + 1

        df0, dg0 = self.dfunFDIPA(w0, x_train, y_train)

        counter[2] = counter[2] + 1

        B0 = self.ddfunFDIPA(x0, L0)
        counter[3] = counter[3] + 1

        n = len(x0)
        m = len(L0)

        E = np.ones([m,1])
        phi = .1 #Multiplica a norma de d1

        epsi = 0.8

        reinicio = 1

        d1 = np.ones_like(x0)

        ###########################################################                                       
        ################### Inicio do Programa ####################
        ###################      FDIPA         ####################
        ###########################################################
        cont = 0
        while cont < tol :
            cont = cont + 1

            counter[0] = counter[0] + 1
            ###########################################################
            ################### Calculo da direção ####################
            ###########################################################    

            if vc == 0:
                M = B0
                vc = 0
            else:
                vc = vc + 1
                M  = np.identity(n)    

            #M = B, B da formula do d0 e d1
            #B00=(M+1*(f0+0*10^(-6))*np.identity(n)).^-1
            #B00 = (M+1*(f0+0*10^(-6))*np.identity(n))
            
            #B00 = B^-1
            B00 = np.linalg.inv(M)

            #L0 = lambda, g0 = G

            # print("div:")
            # print(div)          
            div = (g0/L0 - np.linalg.multi_dot([dg0.T, B00, dg0]))

            BK = B00 + np.linalg.multi_dot([B00, dg0, dg0.T, B00]) / div
            #Primeira direcao d_alpha
            dx1 = -(BK.dot(df0))
            
            if np.linalg.norm(dx1) < 10**(-6):
                # print("saida1")
                # print(cont)
                return self.vet2mat(x0)
            else:
                #Segunda direcao d_beta
                dx2 = (L0/g0)*np.linalg.multi_dot([BK, dg0, E])

                if  (df0.T).dot(dx2) > 0:
                    #rho
                    r0 = min( [phi*np.linalg.norm(dx1)**2, ((epsi-1)*(df0.T).dot(dx1))/(df0.T.dot(dx2))])
                else:
                    r0 = phi*(np.linalg.norm(dx1)**2)               

                #direcao de busca
                dx = dx1 + r0*dx2
                d  = dx
                
                t  = 1 
                xn = x0 + t*dx

                Lx1 = -(L0/g0)*(dg0.T.dot(dx1))
                Lx2 = -(L0/g0)*(dg0.T.dot(dx2)+E)
                L   = np.abs(Lx1+r0*Lx2)

                mat = self.vet2mat(xn)
                fn , gn = self.feedforwardFDIPA(mat, x_train, y_train)
  
                counter[1] = counter[1] + 1

                while ((fn-f0) > 0):
                    t = 0.9 * t
                    xn = x0 + t * dx
                    mat = self.vet2mat(xn)
                    fn , gn = self.feedforwardFDIPA(mat, x_train, y_train)#feedforward
                    counter[1] = counter[1] + 1
                
                ###########################################################
                ################### Criterio de Parada ####################
                ###########################################################
                
                if (np.linalg.norm(f0-fn) < 10**(-6)):
                    # print("saida2")
                    # print(cont)
                    return self.vet2mat(xn)
                
                ###########################################################
                ###################   Parada Forcada   ####################
                ###########################################################

                x00 = x0
                # L00 = L0
                x0  = xn
                
                f0   = fn
                g0   = gn
                df00 = df0
                mat = self.vet2mat(x0)
                df0 , dg0 = self.dfunFDIPA(mat, x_train, y_train)
                dg0 = dg0.reshape(dg0.size,1)
                
                counter[2] = counter[2] + 1
                
                L0 = L + 10**(-8)
                y = df0 - df00
                s = x0 - x00
                B0 = B0 + (y.dot(y.T)) / (y.T.dot(s)) - (np.linalg.multi_dot([B0,s,s.T,B0.T])/(np.linalg.multi_dot([s.T, B0, s])))
        # print("saida3")
        # print(cont)
        return self.vet2mat(xn)
        # return [xn, L, fn, gn, counter, t, d, r0]

    def SGD2(self, X, y, epochs, mini_batch_size, eta, test_data = None):
        n = len(X)
        display_step = 50
        for j in range(epochs):
            p = np.random.permutation(len(X))
            X = X[p]
            y = y[p]
            mini_batches = [(list(zip(X[k:k+mini_batch_size], y[k:k+mini_batch_size]))) for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:            
                self.update_mini_batch(mini_batch, eta)

            if j % display_step == 0:
                predictions = np.asarray([self.feedforward(x) for x in X])
                accuracy = np.sum(((predictions >= 0.5) == y))/X.shape[0]
                print("Epoch {}/{}  | Acurácia {:.4f}".format(j, epochs, accuracy))

# funcao de ativacao sigmoide
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# def sigmoid(z):
#     sig = np.vectorize(sigmoidAux)
#     return sig(z)
    
# def sigmoidAux(z):
#     if z > 20:
#         return 1
#     elif z < -20:
#         return 0
#     else:
#         return 1.0/(1.0 + np.exp(-z))

# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#converte matriz para vetor
def mat2vet(w):
    retorno = np.concatenate(list(map(lambda a: a.reshape(-1), w))) #concatena lista de arrays
    retorno = retorno.reshape(retorno.size, 1)
    return retorno 

rede1 = NeuralNet([2,3,1])