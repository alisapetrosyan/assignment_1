#!/usr/bin/env python
# coding: utf-8

# In[40]:


#trying to make the class uncorruptible by implementing private methods 
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
        if hits < 0 or type(hits) != int: 
            raise ValueError ("Hits cannot be negative or non integer.")
        if misses < 0 or type(misses) != int:
            raise ValueError ("Misses cannot be negative or non integer.")   
        if falseAlarms < 0 or type(falseAlarms) != int:
            raise ValueError ("False Alarms cannot be negative or non integer.")
        if correctRejections < 0 or type(correctRejections) != int:
            raise ValueError ("Correct Rejections cannot be negative or non integer.")
        self.__hits = hits
        self.__misses = misses
        self.__falseAlarms = falseAlarms
        self.__correctRejections = correctRejections
        self.__H = (hits / (hits + misses))
        self.__FA = (falseAlarms / (falseAlarms + correctRejections))

    def d_prime(self):
          return (norm.ppf(self.__H)) - (norm.ppf(self.__FA))
    
    def criterion(self):
        return (-0.5) * ((norm.ppf(self.__H) + norm.ppf(self.__FA)))
    
# altered code to make it non corruptible
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
    
#creating new methods for addition and multiplication
    def __add__(self, other):
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.falseAlarms + other.falseAlarms, self.correctRejections + other.correctRejections)

    def __mul__ (self, other1):
        return SignalDetection(self.hits * other1, self.misses * other1, self.falseAlarms * other1, self.correctRejections * other1)
    
    
#Adding roc plot 
    def plot_roc(self):
        x = []
        y = []
        #append to the list 
        x.append(0)
        y.append(0)        
        x.append(self.FA())
        y.append(self.H())       
        x.append(1)
        y.append(1)
        plt.title("ROC Curve")
        plt.xlabel("False Alarms")
        plt.ylabel("Hits")
        plt.plot(x, y,'o')
        plt.plot(x,y,'-')
        plt.show()
        
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

        

sd = SignalDetection(50, 30, 20, 30)
sd.plot_roc()
sd.plot_sdt(sd.d_prime())



# In[ ]:





# In[ ]:





# In[ ]:




