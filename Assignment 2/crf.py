from __future__ import division
import numpy as np
from scipy.special import logsumexp 
from scipy.optimize import check_grad
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_bfgs
import time

class CRF(object):

    def __init__(self,L,F):
        '''
        This class implements learning and prediction for a conditional random field.

        Args:
            L: a list of label types
            F: the number of features

        Returns:
            None
        '''

        #W_F should have dimension (|L|,F)
        #while W_T should have dimension (|L|,|L|). |L| refers to the
        #number of label types. The value W_T[i,j] refers to the
        #weight on the potential for transitioning from label L[i]
        #to label L[j]. W_F[i,j] refers to the feature potential
        #between label L[i] and feature dimension j.
        self.L = L
        self.F = F
        self.W_F = np.zeros((len(L),F))
        self.W_T = np.zeros((len(L),len(L)))
        #pass

    def get_params(self):
        '''
        Args:
            None

        Returns:
            (W_F,W_T) : a tuple, where W_F and W_T are the current feature
            parameter and transition parameter member variables.
        '''
        return (self.W_F,self.W_T)
        #pass

    def set_params(self, W_F, W_T):
        '''
        Set the member variables corresponding to the parameters W_F and W_T

        Args:
            W_F (numpy.ndarray): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            None

        '''
        self.W_F = W_F
        self.W_T = W_T
        #pass

    def energy(self, Y, X, W_F=None, W_T=None):
        '''
        Compute the energy of a label sequence

        Args:
            Y (list): a list of labels from L of length T.
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            E (float): The energy of Y,X.
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T

        new_Y = [self.L.index(Y[i]) for i in range(0,len(Y))]
        sumofphif = np.sum(np.take(W_F,new_Y,axis=0)*X)
        sumofphit = np.sum(W_T[new_Y[:len(Y)-1],new_Y[1:len(Y)]])
        return -(sumofphif + sumofphit)
        #pass


    def log_Z(self, X, W_F=None, W_T=None):
        '''
        Computes the log partition function for a feature sequence X
        using the parameters W_F and W_T.
        This computation uses the log-space sum-product message
        passing algorithm.

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            log_Z (float): The log of the partition function given X

        '''
        
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        
        L = len(self.L)
        T = X.shape[0]
        
        self.omega = np.zeros((L,T)) #L by T
        self.alpha = np.zeros((L,T)) #L by T
        
        for i in reversed(range(T-1)): 
            #Omega computation
            betaif = np.sum(W_F*X[i+1],axis=1)
            betait = W_T
            self.omega[:,i] = logsumexp(betaif + betait + self.omega[:,i+1],axis=1)
            
            #Alpha computation
            #Use column vectors for alpha computation
            betaifa = np.sum(W_F*X[T-i-2],axis=1,keepdims=True)
            betaita = W_T
            self.alpha[:,T-i-1] =logsumexp(betaifa + betaita + self.alpha[:,T-i-2][:,np.newaxis],axis=0)
            
        #Add in the last node potential for logz    
        self.logz = logsumexp(self.omega[:,0] + np.sum(W_F*X[0],axis=1))
        
        #Sanity check:
        logza = logsumexp(self.alpha[:,T-1] + np.sum(W_F*X[T-1],axis=1))
        #print self.logz,logza
        
        return self.logz
        #pass

    def predict_logprob(self, X, W_F=None, W_T=None):
        '''
        Compute the log of the marginal probability over the label set at each position in a
        sequence of length T given the features in X and parameters W_F and W_T

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

       Returns:
           log_prob (numpy.ndarray): log marginal probabilties, a numpy array of type float w/ dimension (T, |L|)
           log_pairwise_marginals (numpy.ndarray): log marginal probabilties, a numpy array of type float w/ dimension (T - 1, |L|, |L|)
               - log_pairwise_marginals[t][l][l_prime] should represent the log probability of the symbol, l, and the next symbol, l_prime,
                 at time t.
               - Note: log_pairwise_marginals is a 3 dimensional array.
               - Note: the first dimension of log_pairwise_marginals is T-1 because
                       there are T-1 total pairwise marginals in a sequence in length T
        '''
        
        
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        
        self.log_Z(X,W_F,W_T)   
        
        L = len(self.L)
        T = X.shape[0]
        
        #Single node marginals
        self.log_prob = np.zeros((T,L))
        for i in range(T):
            self.log_prob[i] = self.alpha[:,i] + self.omega[:,i] + np.sum(W_F*X[i],axis=1) - self.logz
        #print self.log_prob.shape
        
        #Pairwise marginals
        self.log_pairwise_marginals = np.zeros((T-1,L,L))
        for i in range(T-1):
            #alphal(y) + betal(y,y') + omegal+1(y') + nodepotentiall(y) + nodepotential(y') - logz
            self.log_pairwise_marginals[i] = self.alpha[:,i][:,np.newaxis] + W_T + self.omega[:,i+1] + np.sum(W_F*X[i],axis=1,keepdims=True) + np.sum(W_F*X[i+1],axis=1) - self.logz
        #print self.log_pairwise_marginals.shape
        
        return self.log_prob, self.log_pairwise_marginals
        #pass
        
    def predict(self, X, W_F=None, W_T=None):
        '''
        Return a list of length T containing the sequence of labels with maximum
        marginal probability for each position in an input fearture sequence of length T.

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            Yhat (list): a list of length T containing the max marginal labels given X
        '''
        
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T

        self.predict_logprob(X,W_F,W_T)
        
        Yhatindices = np.argmax(self.log_prob,axis=1)
        Yhat = [self.L[Yhatindices[i]] for i in range(0,len(Yhatindices))]

        assert len(Yhat) == X.shape[0]
        return Yhat

    def log_likelihood(self, Y, X, W_F=None, W_T=None):
        '''
        Calculate the average log likelihood of N labeled sequences under
        parameters W_F and W_T. This must be computed as efficiently as possible.

        Args:
            Y (list): a list of length N where each element n is a list of T_n labels from L
            X (list): a list of length N where each element n is a feature array of shape (T_n,F)
            W_F (numpy.ndarray, optional): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            mll (float): the mean log likelihood of Y and X
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        mll = 0.0
        N = len(Y)
        for i in range(N):
            mll += -self.energy(Y[i],X[i],W_F,W_T) - self.log_Z(X[i],W_F,W_T)
            #print mll
        mll /= N
        return mll

    def gradient_log_likelihood(self, Y, X, W_F=None, W_T=None):
        '''
        Compute the gradient of the average log likelihood
        given the parameters W_F, W_T. Your implementation
        must be as efficient as possible.

        Args:
            Y (list): a list of length N where each element n is a list of T_n labels from L
            X (list): a list of length N where each element n is a feature array of shape (T_n,F)
            W_F (numpy.ndarray, optional): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            (gW_F, gW_T) (tuple): a tuple of numpy arrays the same size as W_F and W_T containing the gradients

        '''
        if W_F is None:
            W_F = self.W_F #C by F
        if W_T is None:
            W_T = self.W_T #C by C
            
        gW_F = np.zeros((W_F.shape)) #C by F
        gW_T = np.zeros((W_T.shape)) #C by C
        
        N = len(Y)
        
        for i in range(N): #Loop over data
            self.predict_logprob(X[i],W_F,W_T)
            for j in range(len(X[i])): #T changes based on number of characters in train image
                yindex = self.L.index(Y[i][j]) #C value for Yij
                gW_F[yindex] = gW_F[yindex] + X[i][j] #Data point update
                #Expectation update for all sum over values of Yk [y'ij=val]*Xnij
                gW_F = gW_F - np.exp(self.log_prob[j])[:,np.newaxis]*X[i][j]
                
            for j in range(len(X[i])-1):
                yindex = self.L.index(Y[i][j]) #C value for Yij
                yindexplusone = self.L.index(Y[i][j+1]) #C' value for Yij+1
                gW_T[yindex][yindexplusone] = gW_T[yindex][yindexplusone] + 1 #Data point update
                #Expectation update for all pairwise marginals
                gW_T = gW_T - np.exp(self.log_pairwise_marginals[j])
            
        gW_F /= N
        gW_T /= N
        
        assert gW_T.shape == W_T.shape
        assert gW_F.shape == W_F.shape

        return (gW_F, gW_T)
    
    def likelihoodwrapper(self,param,Y,X):
        
        C = len(self.L)
        F = self.F
        
        W_F = np.reshape(param[:C*F],(C,F))
        W_T = np.reshape(param[C*F:],(C,C))
        
        mll = self.log_likelihood(Y, X, W_F, W_T)
        #Maximizing the log likelihood is equivalent to minimizing the negative log likelihood
        return -mll
    
    def gradientwrapper(self,param,Y,X):
        
        C = len(self.L)
        F = self.F
        
        W_F = np.reshape(param[:C*F],(C,F))
        W_T = np.reshape(param[C*F:],(C,C))
        
        gW_F,gW_T = self.gradient_log_likelihood(Y, X, W_F, W_T)
        
        return -np.concatenate((gW_F.flatten(),gW_T.flatten()))
    
    def gradcheck(self,Y,X):
        
        param = np.concatenate((self.W_F.flatten(),self.W_T.flatten()))
        print check_grad(self.likelihoodwrapper,self.gradientwrapper,param,Y,X)
        
    def timepredict(self, X, W_F=None, W_T=None):
        #Returns time needed to run predict
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
            
        start = time.time()
        self.predict(X,W_F,W_T)
        stop = time.time()
        #print start,stop
        return stop - start
        

    def fit(self, Y, X):
        '''
        Learns the CRF model parameters W_F, W_F given N labeled sequences as input.
        Sets the member variables W_T and W_F to the learned values

        Args:
            Y (list): a list of length N where each element n is a list of T_n labels from L
            X (list): a list of length N where each element n is a feature array of shape (T_n,F)

        Returns:
            None

        '''
        initialguess = np.concatenate((self.W_F.flatten(),self.W_T.flatten()))
        #flatweights = fmin_bfgs(self.likelihoodwrapper,initialguess,self.gradientwrapper,args=(Y,X))
        returnedlist = fmin_l_bfgs_b(self.likelihoodwrapper,initialguess,self.gradientwrapper,args=(Y,X),pgtol=1e-26,factr=10)
        #print returnedlist
        flatweights = returnedlist[0]
        funcvalatmin = returnedlist[1]
        infodict = returnedlist[2]
        
        print "Optima: ",funcvalatmin
        print "Warning flag (0 implies method has converged): ",infodict['warnflag']
        
        C = len(self.L)
        F = self.F
        
        W_F = np.reshape(flatweights[:C*F],(C,F))
        W_T = np.reshape(flatweights[C*F:],(C,C))
        
        self.set_params(W_F,W_T)
        
def main():
    
    CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
         'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
         'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
         'y', 'z']

    crf = CRF(L=CHARS, F=321)

    def to_list(y):
        y = y.replace("\n", "")
        return [i for i in y]

    Y = [to_list(i) for i in open("../data/train_words.txt")]
    X = [np.loadtxt("../data/train_img{}.txt".format(i),ndmin=2) for i in range(1,len(Y) + 1)]

    X1 = X[:10]
    y1 = Y[:10]
    
    #print X1,y1
    #crf.set_params(W_F=np.random.rand(26,321), W_T=np.random.rand(26,26))
    #crf.set_params(W_F=np.zeros((26,321)), W_T=np.zeros((26,26)))
    #crf.energy(y1,X1)
    #crf.log_Z(X1)
    #crf.predict_logprob(X1)
    #crf.predict(X1)
    #crf.log_likelihood(y1,X1)
    #crf.gradcheck(Y[:1],X[:1])
    #crf.gradient_log_likelihood(Y[:3],X[:3])
    #crf.fit(y1,X1)
    #print crf.W_T
        
    
if __name__ == "__main__":
    main()
