from __future__ import division
import numpy as np
from itertools import *

"""
Variable dict structure
Key = Name of node
Value = List [column number in csv data, dict of values, number of values, description]
"""
"""
variabledict = {'A':[1,{1:'<45',2:'45-55',3:'>=55'},3,'Age'],
                'G':[2,{1:'Female',2:'Male'},2,'Gender'],
                'CP':[3,{1:'Typical',2:'Atypical',3:'Non-Anginal',4:'None'},4,'Chest Pain'],
                'BP':[4,{1:'Low',2:'High'},2,'Blood Pressure'],
                'CH':[5,{1:'Low',2:'High'},2,'Cholesterol'],             
                'ECG':[6,{1:'Normal',2:'Abnormal'},2,'Electrocardiograph'],
                'HR':[7,{1:'Low',2:'High'},2,'Exercise Heart Rate'], 
                'EIA':[8,{1:'No',2:'Yes'},2,'Exercise Induced Angina'], 
                'HD':[9,{1:'No',2:'Yes'},2,'Heart Disease']
                }
"""
variabledict = {'A':[1,{1:'<45',2:'45-55',3:'>=55'},3,'Age'],
                'G':[2,{1:'F',2:'M'},2,'Gender'],
                'CP':[3,{1:'Typical',2:'Atypical',3:'Non-Angial',4:'None'},4,'Chest Pain'],
                'BP':[4,{1:'L',2:'H'},2,'Blood Pressure'],
                'CH':[5,{1:'L',2:'H'},2,'Cholesterol'],             
                'ECG':[6,{1:'Normal',2:'Abnormal'},2,'Electrocardiograph'],
                'HR':[7,{1:'L',2:'H'},2,'Exercise Heart Rate'], 
                'EIA':[8,{1:'N',2:'Y'},2,'Exercise Induced Angina'], 
                'HD':[9,{1:'N',2:'Y'},2,'Heart Disease']
                }


class CPT(object):
    """ 
    This class builds the CPT table for each node in Bayes Net
    Inputs: 
    node - A single node in the graph, key in variabledict (str) 
    listofparents = List containing immediate parents for node in the graph (list of keys in variabledict)
    """ 
    def __init__(self,node,listofparents):
        """
        Initializing class
        """
        #store name of the node (key in variabledict)
        self.nodename = node
        #store column index of node in orginial csv file
        self.nodeindex = variabledict[node][0] - 1 #subtract 1 as array indexing starts from 0
        #np array of categorical values for current node
        self.nodecategories = self.getcategoricalvalues([self.nodename])
        if listofparents:
            #store parent nodes sorted based on column index in original CSV file
            self.listofparents = self.sortparents(listofparents) 
            #store column indices of parents (in sorted order)
            self.listofparentindices = self.getparentindices() #subtract 1 as array indexing starts from 0
            self.listofparentindices = (np.array(self.listofparentindices) - 1).tolist()
            #np array of catergorical values for parents (2D)
            self.parentcategories = self.getcategoricalvalues(self.listofparents)
            #list of categorical values for both node and parents (in that specific order)
            self.categories = self.nodecategories + self.parentcategories
            #np array of all possible permutations of parents
            self.parentpermutation = self.createpermutation(self.parentcategories)
        else:
            self.listofparents = []
            self.listofparentindices = []
            self.parentcategories = []
            self.categories = self.nodecategories
            self.parentpermutation = []
        #np array of all possible permutations of current node and parents if any
        self.permutation = self.createpermutation(self.categories)
        #Create CPT and intialize to uniform distribution over target variable for each setting of parent variables
        self.cpt = self.createCPT()
        
    def createCPT(self):
        """
        Output: Returns the cpt numpy array 
        """
        tempcpt = self.permutation
        tempcpt = np.insert(tempcpt,tempcpt.shape[1],np.zeros(tempcpt.shape[0],dtype='float64'),axis=1)
        tempcpt[:,tempcpt.shape[1]-1] += 1.0/len(self.nodecategories[0])
        #print tempcpt
        return tempcpt
        
    
    def sortparents(self,listofparents):
        """
        Input: Parent list 
        Output: Parent list sorted according to column index in csv file
        """
        index = []
        for parent in listofparents:
            index.append(variabledict[parent][0])
        return [parent for _,parent in sorted(zip(index,listofparents))]
    
    def getparentindices(self):
        """
        Output: Returns column indices of parents in csv data
        """
        index = []
        for parent in self.listofparents:
            index.append(variabledict[parent][0])
        return index
    
    def getcategoricalvalues(self,nodelist):
        categories = []
        for item in nodelist:
            categories.append(variabledict[item][1].keys())
        """
        if len(nodelist) == 1:
            return categories[0]
        else:
            return categories
        """
        return categories
    
    def createpermutation(self,permutationlist):
        permutations = []
        iterable = product(*permutationlist)
        for tupleitem in iter(iterable):
            permutations.append(list(tupleitem))
        return np.array(permutations,dtype='float64')


class BayesNet(object):
    """
    This class implements a Bayes Net
    """
    
    def __init__(self):
        '''
        This method will run upon initialization of the Bayes Net class
        You can structure this class in whatever way seems best.

        The class will need to support four methods by the end of the assignment.
            - fit: sets the parameters of the Bayes Net based on data
            - predict_hd: predicts a heart disease value, based on observed data
            - get: returns a given real-valued parameter
            - set: set the value of a parameter

        input:
            - None
        returns:
            - None
        '''
        self.Acpt = CPT('A',[])
        self.Gcpt = CPT('G',[])
        self.CPcpt = CPT('CP',['HD'])
        self.BPcpt = CPT('BP',['G'])
        self.CHcpt = CPT('CH',['A','G'])
        self.ECGcpt = CPT('ECG',['HD'])
        self.HRcpt = CPT('HR',['A','BP','HD'])
        self.EIAcpt = CPT('EIA',['HD'])
        self.HDcpt = CPT('HD',['CH','BP'])
    
    
    def get(self, target_variable, condition_variables):
        '''
        This method does a lookup of a parameter value in your BayesNet
        For instance, you might want to lookup of p_theta(HD=N | CH=L, BP=L)

        inputs:
            - target_variable and value:
                - a dictionary, such as {'HD':'N'}
            - condition_variables and values
                - a dictionary, such as {'CH':'L', 'BP':'L'}
        returns:
            - The parameter value, a real value within [0,1]
            - If there is a no such parameter in the model, return None
        '''
        allcptobjs = [self.Acpt,self.Gcpt,self.CPcpt,self.BPcpt,self.CHcpt,self.ECGcpt,self.HRcpt,self.EIAcpt,self.HDcpt]
        for cptobj in allcptobjs:
            if True:#try:
                #cptobj.nodename is a string, target_variable.keys() returns a list
                if [cptobj.nodename] == target_variable.keys():
                    relevantcptobj = cptobj
                    break
            else:#except:
                #target_variable must have only a single key
                return None
        if True:#try:
            dictofparticularnode = variabledict[relevantcptobj.nodename][1]
            nodevalue = dictofparticularnode.keys()[dictofparticularnode.values().index(target_variable.values()[0])]
            #print nodevalue
        else:#except:
            return None
        if True:#try:
            parents = condition_variables.keys() #e.g. ['CH','BP']
            sortedparents = relevantcptobj.sortparents(parents)
            #print sortedparents
            parentvalues = []
            for parentnode in sortedparents:
                dictofparticularnode = variabledict[parentnode][1]
                parentnodevalue = dictofparticularnode.keys()[dictofparticularnode.values().index(condition_variables[parentnode])]
                #print parentnodevalue
                parentvalues.append(parentnodevalue)
        else:#except:
            return None
        if True:#try:
            #print relevantcptobj.cpt.shape
            parameterindex = np.where((relevantcptobj.permutation == [nodevalue]+parentvalues).all(axis=1))
            out = relevantcptobj.cpt[parameterindex][0][-1] 
            #print out
        else:#except:
            return None
        return out

    def set(self, target_variable, condition_variables, value):
        '''
        This method sets a parameter value in your BayesNet to value

        After you call the method, the parameter should be set to value
        For instance, you might want to set p(HD|BP,CH) = .222

        inputs:
            - target_variable and value:
                - a dictionary, such as {'HD':'N'}
            - condition_variables and values
                - a dictionary, such as {'CH':'L', 'BP':'L'}
            - value:
                -  probability between 0 and 1
        returns:
            - None
        '''
        allcptobjs = [self.Acpt,self.Gcpt,self.CPcpt,self.BPcpt,self.CHcpt,self.ECGcpt,self.HRcpt,self.EIAcpt,self.HDcpt]
        for cptobj in allcptobjs:
            if True:#try:
                #cptobj.nodename is a string, target_variable.keys() returns a list
                if [cptobj.nodename] == target_variable.keys():
                    relevantcptobj = cptobj
                    break
            else:#except:
                #target_variable must have only a single key
                return None
        if True:#try:
            dictofparticularnode = variabledict[relevantcptobj.nodename][1]
            nodevalue = dictofparticularnode.keys()[dictofparticularnode.values().index(target_variable.values()[0])]
            #print nodevalue
        else:#except:
            return None
        if True:#try:
            parents = condition_variables.keys() #e.g. ['CH','BP']
            sortedparents = relevantcptobj.sortparents(parents)
            #print sortedparents
            parentvalues = []
            for parentnode in sortedparents:
                dictofparticularnode = variabledict[parentnode][1]
                parentnodevalue = dictofparticularnode.keys()[dictofparticularnode.values().index(condition_variables[parentnode])]
                #print parentnodevalue
                parentvalues.append(parentnodevalue)
        else:#except:
            return None
        if True:#try:
            #print relevantcptobj.cpt.shape
            parameterindex = np.where((relevantcptobj.permutation == [nodevalue]+parentvalues).all(axis=1))
            #print relevantcptobj.cpt[parameterindex]
            relevantcptobj.cpt[parameterindex] = [nodevalue]+ parentvalues + [value]
            #print relevantcptobj.cpt[parameterindex] 
        else:#except: 
            return None
        #pass 
    def genericfit(self,data,currentcpt):
        #Handling nodes without parents first
        if currentcpt.listofparents == []:
            #Node does not have parents, MLE is that of multinoulli distribution (Nx/N)
            dataofinterest = np.take(data,[currentcpt.nodeindex],axis=1)
            denom = dataofinterest.shape[0]
            for p in currentcpt.permutation:
                #find number of occurences of permutation p (in this case single categorical value)
                numofp = np.where((dataofinterest == p))[0].shape[0]
                #update parameter value by numofp/denom
                parametermle = numofp/denom
                parameterindex = np.where((currentcpt.permutation == p).all(axis=1))
                currentcpt.cpt[parameterindex] = p.tolist() + [parametermle]            
        else:
            #Node has parents, MLE = Nx,y/Ny
            dataofinterestforparents = np.take(data,currentcpt.listofparentindices,axis=1)
            dataofinterestfornode = np.take(data,[currentcpt.nodeindex] + currentcpt.listofparentindices,axis=1)
            for p in currentcpt.permutation:
                #print p
                #find number of occurences of node + parent permutation in dataofinterestfornode
                #note that np.take returns data in the order of indices 
                #so we do not need to sort node and parents, we can take nodeindex and then parent columns 
                numofp = np.where((dataofinterestfornode == p).all(axis=1))[0].shape[0]
                #find number of occurences of parent permutation in dataofinterestforparents
                denom = np.where((dataofinterestforparents == p[1:]).all(axis=1))[0].shape[0] 
                parametermle = numofp/denom
                #print numofp,denom,parametermle
                parameterindex = np.where((currentcpt.permutation == p).all(axis=1))
                #print currentcpt.cpt[parameterindex]
                currentcpt.cpt[parameterindex] = p.tolist() + [parametermle]
                #print currentcpt.cpt[parameterindex]              
                
    def fit(self, data):
        '''
        This method sets the parameters of your BayesNet to their MLEs
        based on the provided data. The layout of the data array and the
        coding used is described in the handout.

        input:
            - data, a numpy array with the schema described in the handout
        returns:
            - None
        '''
        self.genericfit(data,self.Acpt)
        self.genericfit(data,self.Gcpt)
        self.genericfit(data,self.CPcpt)
        self.genericfit(data,self.BPcpt)
        self.genericfit(data,self.CHcpt)
        self.genericfit(data,self.ECGcpt)
        self.genericfit(data,self.HRcpt)
        self.genericfit(data,self.EIAcpt)
        self.genericfit(data,self.HDcpt)
        #pass
    
    def predict_hd_get(self,currentcpt,datapoint):
        paramsindex = np.where((currentcpt.permutation == np.take(datapoint,[currentcpt.nodeindex]+currentcpt.listofparentindices)).all(axis=1))
        paramvalue = currentcpt.cpt[paramsindex][0][-1]
        #print [currentcpt.nodeindex]+currentcpt.listofparentindices
        #print np.take(datapoint,[currentcpt.nodeindex]+currentcpt.listofparentindices)
        #print paramsindex
        #print currentcpt.cpt[paramsindex]
        #print paramvalue
        return paramvalue
    
    def predict_hd(self, data):
        '''
        - input:
            - data. An array of shape (N,D). The layout of the data array and the
        coding used is described in the handout.

        - returns:
            - the predictions for your data, a numpy array with shape = (N,)
        '''
        
        N,D = data.shape
        predictions = []
        out = np.zeros((N,0))
        #Set heart disease column in the data to Yes(2)
        data = np.delete(data,self.HDcpt.nodeindex,axis=1)
        data = np.insert(data,self.HDcpt.nodeindex,np.full(N,2),axis=1)
        for datapoint in data:
            
            #Heart disease value is 2
            avalfs = self.predict_hd_get(self.Acpt,datapoint) #P(A)
            gvalfs = self.predict_hd_get(self.Gcpt,datapoint) #P(G)
            cpvalfs = self.predict_hd_get(self.CPcpt,datapoint) #P(CP)
            bpvalfs = self.predict_hd_get(self.BPcpt,datapoint) #P(BP)
            chvalfs = self.predict_hd_get(self.CHcpt,datapoint) #P(CH)
            ecgvalfs = self.predict_hd_get(self.ECGcpt,datapoint) #P(ECG)
            hrvalfs = self.predict_hd_get(self.HRcpt,datapoint) #P(HR)
            eiavalfs = self.predict_hd_get(self.EIAcpt,datapoint) #P(EIA)
            hdvalfs = self.predict_hd_get(self.HDcpt,datapoint) #P(HD)
            
            hdfs = avalfs*gvalfs*cpvalfs*bpvalfs*chvalfs*ecgvalfs*hrvalfs*eiavalfs*hdvalfs
            """
            #Set heart disease to second setting and recompute probabilities 
            if datapoint[8] == 1:
                #first setting is low (1)
                firstsettinglow = True
                datapoint[8] = 2
            else:
                firstsettinglow = False
                #fist setting is high (2)
                datapoint[8] = 1
            """
            
            #Set heart disease value to No (1)
            datapoint = np.delete(datapoint,self.HDcpt.nodeindex)
            datapoint = np.insert(datapoint,self.HDcpt.nodeindex,1)
            
            #Heart disease value is 1
            avalss = self.predict_hd_get(self.Acpt,datapoint) #P(A)
            gvalss = self.predict_hd_get(self.Gcpt,datapoint) #P(G)
            cpvalss = self.predict_hd_get(self.CPcpt,datapoint) #P(CP)
            bpvalss = self.predict_hd_get(self.BPcpt,datapoint) #P(BP)
            chvalss = self.predict_hd_get(self.CHcpt,datapoint) #P(CH)
            ecgvalss = self.predict_hd_get(self.ECGcpt,datapoint) #P(ECG)
            hrvalss = self.predict_hd_get(self.HRcpt,datapoint) #P(HR)
            eiavalss = self.predict_hd_get(self.EIAcpt,datapoint) #P(EIA)
            hdvalss = self.predict_hd_get(self.HDcpt,datapoint) #P(HD)                
        
            hdss = avalss*gvalss*cpvalss*bpvalss*chvalss*ecgvalss*hrvalss*eiavalss*hdvalss
            """    
            if firstsettinglow:
                hdhigh = hdss/(hdfs+hdss)
            else:
                hdhigh = hdfs/(hdfs+hdss)
            """
            hdhigh = hdfs/(hdfs+hdss)
            #print hdhigh
            if hdhigh < 0.5:
                predictions.append(1)
            else:
                predictions.append(2) 
                
        #print predictions
        #print np.array(predictions).shape       
        return np.array(predictions,dtype='int')

def main():
    mybayesnet = BayesNet()
    """
    print mybayesnet.HRcpt.nodename, mybayesnet.HRcpt.listofparents
    for item in mybayesnet.HRcpt.permutation:
        print item
    """
    mybayesnet.get({'HD':'N'},{'CH':'L', 'BP':'L'})
    mybayesnet.set({'HD':'N'},{'CH':'L', 'BP':'L'},0.222)
    data = np.loadtxt('../data/data-train-1.txt',delimiter = ',')
    #print data.shape
    mybayesnet.fit(data)
    
    chhighnum = mybayesnet.get({'A':'45-55'},{})*mybayesnet.get({'G':'M'},{})*mybayesnet.get({'CP':'None'},{'HD':'N'})*mybayesnet.get({'BP':'L'},{'G':'M'})*mybayesnet.get({'ECG':'Normal'},{'HD':'N'})
    chhighnum *= mybayesnet.get({'HR':'L'},{'A':'45-55','BP':'L','HD':'N'})*mybayesnet.get({'EIA':'N'},{'HD':'N'})*mybayesnet.get({'HD':'N'},{'CH':'H','BP':'L'})*mybayesnet.get({'CH':'H'},{'A':'45-55','G':'M'})
    
    chlownum = mybayesnet.get({'A':'45-55'},{})*mybayesnet.get({'G':'M'},{})*mybayesnet.get({'CP':'None'},{'HD':'N'})*mybayesnet.get({'BP':'L'},{'G':'M'})*mybayesnet.get({'ECG':'Normal'},{'HD':'N'})
    chlownum *= mybayesnet.get({'HR':'L'},{'A':'45-55','BP':'L','HD':'N'})*mybayesnet.get({'EIA':'N'},{'HD':'N'})*mybayesnet.get({'HD':'N'},{'CH':'L','BP':'L'})*mybayesnet.get({'CH':'L'},{'A':'45-55','G':'M'})
    
    phigh = chhighnum/(chhighnum + chlownum)
    plow = chlownum/(chhighnum + chlownum)
    
    print phigh,plow,phigh+plow
    
    bn = BayesNet()
    bn.fit(data)
    
    bphighgendermale = bn.get({'A':'45-55'},{})*bn.get({'G':'M'},{})*bn.get({'CP':'Typical'},{'HD':'N'})*bn.get({'BP':'H'},{'G':'M'})*bn.get({'ECG':'Normal'},{'HD':'N'})
    bphighgendermale *= bn.get({'HR':'H'},{'A':'45-55','BP':'H','HD':'N'})*bn.get({'EIA':'Y'},{'HD':'N'})*bn.get({'HD':'N'},{'CH':'H','BP':'H'})*bn.get({'CH':'H'},{'A':'45-55','G':'M'})

    bphighgenderfemale = bn.get({'A':'45-55'},{})*bn.get({'G':'F'},{})*bn.get({'CP':'Typical'},{'HD':'N'})*bn.get({'BP':'H'},{'G':'F'})*bn.get({'ECG':'Normal'},{'HD':'N'})
    bphighgenderfemale *= bn.get({'HR':'H'},{'A':'45-55','BP':'H','HD':'N'})*bn.get({'EIA':'Y'},{'HD':'N'})*bn.get({'HD':'N'},{'CH':'H','BP':'H'})*bn.get({'CH':'H'},{'A':'45-55','G':'F'})

    bplowgendermale = bn.get({'A':'45-55'},{})*bn.get({'G':'M'},{})*bn.get({'CP':'Typical'},{'HD':'N'})*bn.get({'BP':'L'},{'G':'M'})*bn.get({'ECG':'Normal'},{'HD':'N'})
    bplowgendermale *= bn.get({'HR':'H'},{'A':'45-55','BP':'L','HD':'N'})*bn.get({'EIA':'Y'},{'HD':'N'})*bn.get({'HD':'N'},{'CH':'H','BP':'L'})*bn.get({'CH':'H'},{'A':'45-55','G':'M'})
    
    bplowgenderfemale = bn.get({'A':'45-55'},{})*bn.get({'G':'F'},{})*bn.get({'CP':'Typical'},{'HD':'N'})*bn.get({'BP':'L'},{'G':'F'})*bn.get({'ECG':'Normal'},{'HD':'N'})
    bplowgenderfemale *= bn.get({'HR':'H'},{'A':'45-55','BP':'L','HD':'N'})*bn.get({'EIA':'Y'},{'HD':'N'})*bn.get({'HD':'N'},{'CH':'H','BP':'L'})*bn.get({'CH':'H'},{'A':'45-55','G':'F'})

    bphigh = (bphighgendermale + bphighgenderfemale)/(bphighgendermale + bphighgenderfemale + bplowgendermale + bplowgenderfemale)
    bplow = (bplowgendermale + bplowgenderfemale)/(bphighgendermale + bphighgenderfemale + bplowgendermale + bplowgenderfemale)
    
    print bphigh,bplow,bphigh+bplow  
    #print bn.CPcpt.cpt
    print bn.predict_hd(data)
    
if __name__ == "__main__":
    main()