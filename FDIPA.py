import numpy as np

#Todo : Passar
def  FDIPA(self, fun, dfun, ddfun, x0, L0, tol, mini_batch):
    ################################################################
    ###################      Dados iniciais     ####################
    ################################################################
    counter = np.zeros([1,5])
    Buscatol = 0
    vc = 0

    f0 , g0 = fun(mini_batch, x0)
    counter[1] = counter[1] + 1

    df0, dg0 = dfun(mini_batch, x0)
    counter[2] = counter[2] + 1

    B0 = ddfun(x0,L0)
    counter[3] = counter[3] + 1

    n = len(x0)
    m = len(L0)

    E = np.ones([m,1])
    phi = .1 #Multiplica a norma de d1

    epsi = 0.8

    siga = 1
    reinicio = 1

    d1 = np.ones_like(x0)

    ###########################################################                                       
    ################### Inicio do Programa ####################
    ###################      FDIPA         ####################
    ###########################################################
    while siga:
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
        BK = B00 + np.linalg.multi_dot([B00, dg0, dg0.T,B00]) / (g0/L0 - np.linalg.multi_dot([dg0.T, B00, dg0])) 
        #Primeira direcao d_alpha
        dx1 = -(BK.dot(df0))
        
        if np.linalg.norm(dx1) < 10**(-16):
            siga = 0
            r0   = 0  
            xn   = x0
            fn   = f0
            d    = d1
            #L    = L1
            gn   = g0
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
            
            Lx1 = -(L0/g0)*dg0.T.dot(dx1)
            Lx2 = -(L0/g0)*(dg0.T.dot(dx2)+E)
            L   = np.abs(Lx1+r0*Lx2)

            fn , gn = fun(mini_batch, xn)
            counter[1] = counter[1] + 1

            while ((fn-f0) > 0):
                t = 0.9 * t
                xn = x0 + t * dx
                fn , gn = fun(mini_batch, xn) #feedforward
                counter[1] = counter[1] + 1
            
            ###########################################################
            ################### Criterio de Parada ####################
            ###########################################################
            
            siga = (np.linalg.norm(f0-fn) > tol)
            
            ###########################################################
            ###################   Parada Forcada   ####################
            ###########################################################

            if  counter[0] > 200:
                siga = 0

            x00 = x0
            L00 = L0
            x0  = xn
            
            f0   = fn
            g0   = gn
            df00 = df0
            df0 , dg0 = dfun(mini_batch, x0)
            counter[2] = counter[2] + 1
            
            L0 = L + 10**(-8)
            y = df0 - df00
            s = x0 - x00
            B0 = B0 + (y.dot(y.T)) / (y.T.dot(s)) - (np.linalg.multi_dot([B0,s,s.T,B0.T])/(np.linalg.multi_dot([s.T, B0, s])))

    return [xn, L, fn, gn, counter, t, d, r0]