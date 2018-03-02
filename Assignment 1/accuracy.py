from __future__ import division
from bn import BayesNet
from bn_custom import BayesNetCustom
import numpy as np

train_data_1 = np.loadtxt('../data/data-train-1.txt',delimiter = ',')
train_data_2 = np.loadtxt('../data/data-train-2.txt',delimiter = ',')
train_data_3 = np.loadtxt('../data/data-train-3.txt',delimiter = ',')
train_data_4 = np.loadtxt('../data/data-train-4.txt',delimiter = ',')
train_data_5 = np.loadtxt('../data/data-train-5.txt',delimiter = ',')

test_data_1 = np.loadtxt('../data/data-test-1.txt',delimiter = ',')
test_data_2 = np.loadtxt('../data/data-test-2.txt',delimiter = ',')
test_data_3 = np.loadtxt('../data/data-test-3.txt',delimiter = ',')
test_data_4 = np.loadtxt('../data/data-test-4.txt',delimiter = ',')
test_data_5 = np.loadtxt('../data/data-test-5.txt',delimiter = ',')

sets = [[train_data_1,test_data_1],[train_data_2,test_data_2],[train_data_3,test_data_3],[train_data_4,test_data_4],[train_data_5,test_data_5]]
foldaccuracies = []

for item in sets:
    mybayesnet = BayesNet()
    mybayesnet.fit(item[0])
    predicted_y = mybayesnet.predict_hd(item[1])
    #item[1][:,8] is [[]] and predicted_y is []
    accuracy = np.mean([predicted_y] == item[1][:,8])
    #print accuracy
    foldaccuracies.append(accuracy)

foldaccuraciescustom = []

for item in sets:
    mycustombayesnet = BayesNetCustom()
    mycustombayesnet.fit(item[0])
    predicted_y = mycustombayesnet.predict_hd(item[1])
    #item[1][:,8] is [[]] and predicted_y is []
    accuracy = np.mean([predicted_y] == item[1][:,8])
    #print accuracy
    foldaccuraciescustom.append(accuracy)

print "Regular Bayes Net"   
print "Mean:",np.mean(foldaccuracies)
print "Standard Deviation:",np.std(foldaccuracies)

print "Custom Bayes Net"
print "Mean",np.mean(foldaccuraciescustom)
print "Standard Deviation:",np.std(foldaccuraciescustom)