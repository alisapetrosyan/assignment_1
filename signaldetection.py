#!/usr/bin/env python
# coding: utf-8

# In[47]:


import unittest
import numpy as np
import scipy 
from scipy import stats
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm



#trying to make the class uncorruptible by implementing private methods 

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
    
    def __add__(self, other):
        return SignalDetection(self.__hits + other.__hits, self.__misses + other.__misses, self.__falseAlarms + other.__falseAlarms, self.__correctRejections + other.__correctRejections)

    def __mul__ (self, other1):
        return SignalDetection(self.__hits * other1, self.__misses * other1, self.__falseAlarms * other1, self.__correctRejections * other1)

    def plot_roc(self, hits, falseAlarms):
        return metrics.plot_roc_curve(hits, falseAlarms) 
        plt.show()
        
    def plot_sdt(self):
        
        
        

# tests 
class TestSignalDetection(unittest.TestCase):

    def test_d_prime_zero(self):
        sd   = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_d_prime_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_zero(self):
        sd   = SignalDetection(5, 5, 5, 5)
        # Calculate expected criterion
        expected = 0
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        # Calculate expected criterion
        expected = -0.463918426665941
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_addition(self):
        sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
        expected = SignalDetection(3, 2, 3, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)

    def test_multiplication(self):
        sd = SignalDetection(1, 2, 3, 1) * 4
        expected = SignalDetection(4, 8, 12, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)
        
    def test_d_prime_nonzero_2(self):
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        hits = 100
        misses = 100
        #sd.
        #sd.__criterion = 100
        #sd.__d_prime = 150
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=6)

if __name__ == '__main__':
    unittest.main(argv= ['first-arg-is-ignored'], exit = False)

# test                               
# object 
# check what the d prime is
# inside object 
# change hits 
# see what output you get 
# commit everything alone

# `plots 
# have a point at 0,0 and 1,1
# write method that it can input muliple objects to take it mulitple points on the 
# false alarm on horizontal axis 
# plot roc should take no inputs 
# extract the hit and false alarms 
# take an array 
# (sd, sd1).plot_roc()
# can make a loop and store it in some vector and make the plot 

# iterating a class 
#....(self):
# L = len(self)
# for c in range(L):
# s = self(c)
#sort function?
    


# In[258]:


import unittest
import numpy as np
import scipy 
from scipy import stats
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm



#trying to make the class uncorruptible by implementing private methods 
# changed H and FA to be their own functions instead of inside the initializing function

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

sd = SignalDetection(50, 30, 20, 30)
sd.plot_roc()


# tests 
class TestSignalDetection(unittest.TestCase):

    def test_d_prime_zero(self):
        sd   = SignalDetection(1,1,1,1)
        expected = 0
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_d_prime_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_zero(self):
        sd   = SignalDetection(5, 5, 5, 5)
        # Calculate expected criterion
        expected = 0
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        # Calculate expected criterion
        expected = -0.463918426665941
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_addition(self):
        sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
        expected = SignalDetection(3, 2, 3, 4).criterion()
        obtained = sd.criterion()
        #Compare calculated and expected criterion
        self.assertEqual(obtained, expected)

    def test_multiplication(self):
        sd = SignalDetection(1, 2, 3, 1) * 4
        expected = SignalDetection(4, 8, 12, 4).criterion()
        obtained = sd.criterion()
        #Compare calculated and expected criterion
        self.assertEqual(obtained, expected)
        
    def test_corrption(self):
        sd = SignalDetection(15, 10, 15, 5)
        expected = sd.criterion()
        sd.hits = 100
        obtained2 = sd.criterion()
        self.assertNotEqual(expected, obtained2)

if __name__ == '__main__':
    unittest.main(argv= ['first-arg-is-ignored'], exit = False)

# test                               
# object 
# check what the d prime is
# inside object 
# change hits 
# see what output you get 
# commit everything alone

#plots 
    


# In[ ]:





# In[ ]:




