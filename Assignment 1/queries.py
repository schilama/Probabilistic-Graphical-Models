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
def query_5_a(bn):
    '''
    Please write a method which returns the distribution over the query
    variable, CH.

    You should return a dictionary, {"H":p1, "L":p2} where p1 and p2 are
    the probabilities that the patient has high or low cholesterol.

    You should be able to use the get method from your BayesNet class to implement
    this function.

    inputs:
        - bn: a parameterized Bayes Net
    returns:
        - out: a dictionary of probabilities, {"H":p1, "L":p2}

    '''
    
    chhighnum = bn.get({'A':'45-55'},{})*bn.get({'G':'M'},{})*bn.get({'CP':'None'},{'HD':'N'})*bn.get({'BP':'L'},{'G':'M'})*bn.get({'ECG':'Normal'},{'HD':'N'})
    chhighnum *= bn.get({'HR':'L'},{'A':'45-55','BP':'L','HD':'N'})*bn.get({'EIA':'N'},{'HD':'N'})*bn.get({'HD':'N'},{'CH':'H','BP':'L'})*bn.get({'CH':'H'},{'A':'45-55','G':'M'})
    
    chlownum = bn.get({'A':'45-55'},{})*bn.get({'G':'M'},{})*bn.get({'CP':'None'},{'HD':'N'})*bn.get({'BP':'L'},{'G':'M'})*bn.get({'ECG':'Normal'},{'HD':'N'})
    chlownum *= bn.get({'HR':'L'},{'A':'45-55','BP':'L','HD':'N'})*bn.get({'EIA':'N'},{'HD':'N'})*bn.get({'HD':'N'},{'CH':'L','BP':'L'})*bn.get({'CH':'L'},{'A':'45-55','G':'M'})
    
    phigh = chhighnum/(chhighnum + chlownum)
    plow = chlownum/(chhighnum + chlownum)
    
    out = {"H": phigh, "L": 0}
    out['H'] = phigh
    out['L'] = plow

    return out


def query_5_b(bn):
    '''
    Please write a method which returns an answer to query 5b from the problem set
    input:
        - bn: a parameterized Bayes Net

    returns:
        answers, a dictionary with two keys, "H" and "L". "H" is the probability
        of high BP given the specified conditions. "L" is the probability
        of low BP, given the specified conditions
    '''
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
    
    out = {"H": 0, "L": 0}
    out['H'] = bphigh
    out['L'] = bplow

    return out
