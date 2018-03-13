'''
Use this file to answer question 5.1 and 5.2
'''
from __future__ import division
from crf import CRF
import numpy as np
import matplotlib.pyplot as plt
import timeit

def five_one():
    '''implement your experiments for question 5.1 here'''
    """
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
    
    Y_test = [to_list(i) for i in open("../data/test_words.txt")]
    X_test = [np.loadtxt("../data/test_img{}.txt".format(i),ndmin=2) for i in range(1,len(Y_test) + 1)]
    
    W_Flist = []
    W_Tlist = []
    for i in range(100,900,100):
        X_train = X[:i]
        Y_train = Y[:i]
        crf.fit(Y_train,X_train)
        W_Flist.append(crf.W_F)
        W_Tlist.append(crf.W_T)
    np.save('EightModelsW_F.npy',np.array(W_Flist))
    np.save('EightModelsW_T.npy',np.array(W_Tlist))
    
    """
    """
    EMW_F = np.load('EightModelsW_F.npy')
    EMW_T = np.load('EightModelsW_T.npy')
    #print EMW_F.shape,EMW_T.shape
    
    avgtestlll = []
    avgtestprederr = []
    for i in range(len(EMW_F)):
        crf.set_params(EMW_F[i],EMW_T[i])
        avgtestlll.append(crf.log_likelihood(Y_test,X_test))
        numerr = 0
        totalchars = 0
        for j in range(len(Y_test)):
            numerr += np.sum(np.array(crf.predict(X_test[j]))!=np.array(Y_test[j]))
            totalchars += len(Y_test[j])
        print numerr, totalchars
        avgtestprederr.append(numerr/totalchars)
        #pass
    np.save('AvgTestLLL.npy',np.array(avgtestlll))
    np.save('AvgTestPredErr.npy',np.array(avgtestprederr))
    """
    avgllls = np.load('AvgTestLLL.npy')
    avgerrs = np.load('AvgTestPredErr.npy')

    Xaxis = [i for i in range(100,900,100)]
    
    plt.plot(Xaxis,avgllls,color='green')
    plt.title('Figure 1')
    plt.xlabel('Number of Training Cases')
    plt.ylabel('Average Test Set Conditional Log Likelihood')
    
    plt.show()
    
    plt.plot(Xaxis,avgerrs,color='blue')
    plt.title('Figure 2')
    plt.xlabel('Number of Training Cases')
    plt.ylabel('Average Test Set Prediction Error')
    
    plt.show()

def five_two():
    '''implement your experiments for question 5.2 here'''
    
    W_F = np.load('EightModelsW_F.npy')[7]
    W_T = np.load('EightModelsW_T.npy')[7]
    
    CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
         'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
         'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
         'y', 'z']

    crf = CRF(L=CHARS, F=321)
    crf.set_params(W_F,W_T)

    def to_list(y):
        y = y.replace("\n", "")
        return [i for i in y]
    
    Y_test = [to_list(i) for i in open("../data/test_words.txt")]
    X_test = [np.loadtxt("../data/test_img{}.txt".format(i),ndmin=2) for i in range(1,len(Y_test) + 1)]
    
    #Sort Y_test and X_test based on sequence length of Y_test 
    returnedtuple = zip(*sorted(zip(Y_test,X_test),key=lambda y: len(y[0])))
    Y_test, X_test = returnedtuple[0],returnedtuple[1] 
    
    maxlen = 1
    indiceslist = []
    #Store indices of Y_test when sequence length increases by 1 in the sorted list
    for i in range(len(Y_test)):
        if len(Y_test[i]) == maxlen:
            indiceslist.append(i)
            maxlen += 1
    maxlen -= 1
    
    Y_timeit = []
    X_timeit = []
    #Add the items to Y_timeit and X_timeit
    for i in indiceslist:
        Y_timeit.append(Y_test[i])
        X_timeit.append(X_test[i])
    
    i = len(Y_timeit)-1 #i is the index of the last element (longest sequence) in Y_timeit
    j = 0    #j is the first element (sequence length one in Y_test)
    
    while (maxlen < 20):
        Y_timeit.append(Y_timeit[i] + Y_timeit[j])
        X_timeit.append(np.concatenate((X_timeit[i],X_timeit[j])))
        j += 1
        maxlen += 1
    
    #for i in range(len(Y_timeit)):
        #print Y_timeit[i],len(X_timeit[i])
    seqtimes = []
    for i in range(len(Y_timeit)):
        avgthislist = []
        for j in range(1000):
            avgthislist.append(crf.timepredict(X_timeit[i]))
        seqtimes.append(np.average(avgthislist))
      
    Xaxis = [i for i in range(1,21)]
    plt.plot(Xaxis,seqtimes,color='red')
    plt.title('Max Marginal Inference Time Analyis')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time Taken For Max Marginal Inference (seconds)')
    plt.show()
    #pass

five_one()
five_two()
