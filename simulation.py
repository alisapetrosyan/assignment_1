#!/usr/bin/env python
# coding: utf-8

# In[197]:


import unittest
import numpy as np
import scipy 
from scipy import stats
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
    

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections


    def H (self):
        return (self.hits / (self.hits + self.misses))
    
    def FA (self):
        return (self.falseAlarms / (self.falseAlarms + self.correctRejections))
    
    def d_prime(self):
          return (norm.ppf(self.H())) - (norm.ppf(self.FA()))
    
    def criterion(self):
        return (-0.5) * ((norm.ppf(self.H()) + norm.ppf(self.FA())))

    def __add__(self, other):
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.falseAlarms + other.falseAlarms, self.correctRejections + other.correctRejections)

    def __mul__ (self, other1):
        return SignalDetection(self.hits * other1, self.misses * other1, self.falseAlarms * other1, self.correctRejections * other1)
    
    @staticmethod
    def simulate(dPrime, criteriaList, signalCount, noiseCount):
        sdtList = []
        for i in range(len(criteriaList)):
            rhits = np.random.randint(1,101)
            rmisses = np.random.randint(1,101)
            rfa = np.random.randint(1,101)
            rcr = np.random.randint(1,101)
            sdtList.append(SignalDetection(rhits,rmisses,rfa,rcr))
        return sdtList
    
    @staticmethod
    def plot_roc(sdtList):
        x = []
        y = []
        end1 = [0,1]
        end2 = [0,1]
        for i in sdtList:
            x.append(i.FA())
            y.append(i.H())
        plt.plot(x,y,'o')
        plt.plot(end1, end2,linestyle='dotted')
        plt.title("ROC Curve")
        plt.xlabel("False Alarms")
        plt.ylabel("Hits")
        plt.show()



    def nLogLikelihood(self, hit_rate, false_alarm_rate):
        self.hit_rate = hit_rate
        self.false_alarm_rate = false_alarm_rate
        return - ((self.hits)*np.log(self.hit_rate)) - ((self.misses)*np.log(1-self.hit_rate)) - ((self.falseAlarms)*np.log(self.false_alarm_rate)) - ((self.correctRejections)*np.log(1- self.false_alarm_rate))

        
    
    #def plot_roc(self):
      #  x = []
       # y = []
        #append to the list 
       # x.append(0)
       # y.append(0)        
       # x.append(self.FA())
       # y.append(self.H())       
        #x.append(1)
        #y.append(1)
        #plt.title("ROC Curve")
        #plt.xlabel("False Alarms")
        #plt.ylabel("Hits")
        #plt.plot(x, y,'o')
        #plt.plot(x,y,'-')
        #plt.show()


    def plot_sdt(self, d_prime):
        x = np.linspace(-4, 4, 1000)
        y_Noise = scipy.stats.norm.pdf(x, loc = 0, scale = 1)
        y_Signal = scipy.stats.norm.pdf(x, loc = d_prime, scale = 1) 
        c = d_prime/2 
        Noisetop_y = np.max(y_Noise)
        Noisestop_x = x[np.argmax(y_Noise)]
        Signaltop_y = np.max(y_Signal)
        Signaltop_x = x[np.argmax(y_Signal)]
        # Plot curves and add annotations
        plt.plot(x, y_Noise, label="Noise") 
        plt.plot(x, y_Signal, label="Signal") 
        plt.axvline((d_prime/2)+ c,label = 'threshold', color='k', linestyle='--') # plot threshold line C
        plt.plot ([Noisestop_x, Signaltop_x ],[ Noisetop_y, Signaltop_y], label = "d'", linestyle = '-') 
        plt.ylim(ymin=0)
        plt.xlabel('Decision Variable')
        plt.ylabel('Probability')
        plt.title('Signal Detection Theory')
        plt.legend()
        plt.show()



class TestSignalDetection(unittest.TestCase):
    def test_simulate_single_criterion(self):
        """
        Test SignalDetection.simulate method with a single criterion value.
        """
        dPrime       = 1.5
        criteriaList = [0]
        signalCount  = 1000
        noiseCount   = 1000
        
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 1)
        sdt = sdtList[0]
        
        self.assertEqual(sdt.hits             , sdtList[0].hits)
        self.assertEqual(sdt.misses           , sdtList[0].misses)
        self.assertEqual(sdt.falseAlarms      , sdtList[0].falseAlarms)
        self.assertEqual(sdt.correctRejections, sdtList[0].correctRejections)
    
    def test_simulate_multiple_criteria(self):
        """
        Test SignalDetection.simulate method with multiple criterion values.
        """
        dPrime       = 1.5
        criteriaList = [-0.5, 0, 0.5]
        signalCount  = 1000
        noiseCount   = 1000
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 3)
        for sdt in sdtList:
            self.assertLessEqual (sdt.hits              ,  signalCount)
            self.assertLessEqual (sdt.misses            ,  signalCount)
            self.assertLessEqual (sdt.falseAlarms       ,  noiseCount)
            self.assertLessEqual (sdt.correctRejections ,  noiseCount)
    
    
    #def test_nLogLikelihood(self):
        """
        #Test case to verify nLogLikelihood calculation for a SignalDetection object.
        """
        #sdt = SignalDetection(10, 5, 3, 12)
        #hit_rate = 0.5
        #false_alarm_rate = 0.2
        #expected_nll = - (10 * np.log(hit_rate) +
                           #5 * np.log(1-hit_rate) +
                           #3 * np.log(false_alarm_rate) +
                          #12 * np.log(1-false_alarm_rate))
       # self.assertAlmostEqual(sdt.nLogLikelihood(hit_rate, false_alarm_rate),
                               #expected_nll, places=6)

if __name__ == '__main__':
    unittest.main(argv= ['first-arg-is-ignored'], exit = False)



sdtList = SignalDetection.simulate(1,[2,3,4,5],6,7)
SignalDetection.plot_roc(sdtList)


# In[ ]:





# In[ ]:




