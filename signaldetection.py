#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[45]:


import unittest
import numpy as np
import scipy 
from scipy import stats
from scipy.stats import norm
import math
import matplotlib.pyplot as plt


class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
        self.H = (hits / (hits + misses))
        self.FA = (falseAlarms / (falseAlarms + correctRejections))

    def d_prime(self):
          return (norm.ppf(self.H)) - (norm.ppf(self.FA))
    
    def criterion(self):
        return (-0.5) * ((norm.ppf(self.H) + norm.ppf(self.FA)))
                              


# In[ ]:





# In[ ]:




