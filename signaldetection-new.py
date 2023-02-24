#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
    

        
if __name__ == '__main__':
    unittest.main(argv= ['first-arg-is-ignored'], exit = False)


# In[ ]:





# In[ ]:





# In[ ]:




