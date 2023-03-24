#!/usr/bin/env python
# coding: utf-8

# In[3]:


import unittest
import scipy.stats as stats
import numpy as np
import scipy
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.__hits = hits 
        self.__misses = misses
        self.__falseAlarms = falseAlarms 
        self.__correctRejections = correctRejections

    def hit_rate(self):
        return self.__hits / (self.__hits + self.__misses)

    def falseAlarm_rate(self):
        return self.__falseAlarms / (self.__falseAlarms + self.__correctRejections)

    def d_prime(self):
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        z_hit = stats.norm.ppf(hit_rate) 
        z_falseAlarm = stats.norm.ppf(falseAlarm_rate)
        return z_hit-z_falseAlarm

    def criterion(self):
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        z_hit = stats.norm.ppf(hit_rate)
        z_falseAlarm = stats.norm.ppf(falseAlarm_rate)
        return -0.5*(z_hit + z_falseAlarm)

    def __add__(self, other):
        return SignalDetection(self.__hits + other.__hits, self.__misses + other.__misses, 
                               self.__falseAlarms + other.__falseAlarms, 
                               self.__correctRejections + other.__correctRejections)
    
    def __mul__(self, scalar):
        return SignalDetection(self.__hits * scalar, self.__misses * scalar, 
                               self.__falseAlarms * scalar, 
                               self.__correctRejections * scalar)

    def plot_roc_old(self):
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        plt.plot([0, falseAlarm_rate, 1], [0, hit_rate, 1], 'b')
        plt.scatter(falseAlarm_rate, hit_rate, c='r')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Alarm rate')
        plt.ylabel('Hit rate')
        plt.title('Receiver Operating Characteristic (ROC) curve')
        #plt.show()

    def plot_sdt(self):
        hit_mu = 0
        hit_sigma = 1
        fa_mu = self.d_prime()
        fa_sigma = 1
        x = np.linspace(-4, 4, 500)
        hit_pdf = stats.norm.pdf(x, hit_mu, hit_sigma)
        fa_pdf = stats.norm.pdf(x, fa_mu, fa_sigma) 
        fig, ax = plt.subplots()
        ax.plot(x, hit_pdf, 'r', label='Signal')
        ax.plot(x, fa_pdf, 'g', label='Noise')
        ax.axvline((self.d_prime()/2)+self.criterion(), linestyle='--', color='b', label = "criterion")
        ax.hlines(max(hit_pdf), self.d_prime(), 0, colors='k', linestyles='dashed', label = "d-prime")
        ax.set_xlabel('Evidence')
        ax.set_ylabel('Probability Density')
        ax.set_title('Signal Detection Theory (SDT) Plot')
        ax.legend()

    @staticmethod
    def simulate(dprime, criteriaList, signalCount, noiseCount):
        sdtList = []
        for i in range(len(criteriaList)):
            criteria = criteriaList[i]
            K = criteria + dprime/2
            hit = [1 - norm.cdf(K - dprime)]
            fA = [1 - norm.cdf(K)]
            hits = np.random.binomial(n = signalCount, p = hit)
            misses = signalCount - hits
            falseAlarms = np.random.binomial(n = noiseCount, p = fA)
            correctRejections = noiseCount - falseAlarms
            sdt = SignalDetection(hits, misses, falseAlarms, correctRejections)
            sdt.hits = hits
            sdt.misses = misses
            sdt.falseAlarms = falseAlarms
            sdt.correctRejections = correctRejections
            sdtList.append(sdt)
        return sdtList

    @staticmethod
    def plot_roc(sdtList):
        #plt.figure()
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('false alarm rate')
        plt.ylabel('hit rate')
        plt.title('ROC curve')
        for sdt in sdtList:
            hit_rate = sdt.hit_rate()
            falseAlarm_rate = sdt.falseAlarm_rate()
            plt.plot(falseAlarm_rate, hit_rate, 'ko')

 
    def nLogLikelihood(self, hit_rate, falseAlarm_rate):
        miss_rate = 1 - hit_rate
        correctRejection_rate = 1 - falseAlarm_rate
        likelihood = ((hit_rate ** self.__hits) * (miss_rate ** self.__misses) * 
                      (falseAlarm_rate ** self.__falseAlarms) * 
                      (correctRejection_rate ** self.__correctRejections))
        ell = -np.log(likelihood)
        return ell
    

    @staticmethod
    def rocCurve(falseAlarm_rate, a):
        return norm.cdf(a + norm.ppf(falseAlarm_rate))


    @staticmethod
    def fit_roc(sdtList):
        hit_rates = []
        falseAlarm_rates = []
        plt.plot([0,1], [0,1], 'k--')
        for sdt in sdtList:
            hit_rate = sdt.hit_rate()
            falseAlarm_rate = sdt.falseAlarm_rate()
            hit_rates.append(hit_rate)
            falseAlarm_rates.append(falseAlarm_rate)
            plt.plot(falseAlarm_rate, hit_rate, 'ko')
        # fitting the function: finding value of 'a' that minimizes loss
        def loss(a):
            sumOfSquares = 0
            for i in range(len(hit_rates)):
                p_rate = SignalDetection.rocCurve(falseAlarm_rates[i], a)
                sumOfSquares += (p_rate - hit_rates[i])**2
            return sumOfSquares
        result = minimize(loss, [0])
        a = result.x[0]
        # plot the ROC curve with the fitted curve
        x = np.linspace(0, 1, num=100)
        y = SignalDetection.rocCurve(x, a)
        plt.plot(x, y, 'r-')
        plt.xlabel('false alarm rate')
        plt.ylabel('hit rate')
        plt.title('ROC curve')
        #plt.show()
        return a


    @staticmethod
    def rocLoss(a, sdtList):
        loss = 0.0
        for sdt in sdtList:
            falseAlarm_rate = sdt.falseAlarm_rate()
            hit_rate = SignalDetection.rocCurve(falseAlarm_rate, a) 
            loss += sdt.nLogLikelihood(hit_rate, falseAlarm_rate)
        return loss


sdtList = []
sdtList.append(SignalDetection(11, 1, 15, 5))
sdtList.append(SignalDetection(14, 1, 7, 5))
sdtList.append(SignalDetection(11, 5, 5, 10))
sdtList.append(SignalDetection(8, 5, 1, 5))
sdtList.append(SignalDetection(17, 3, 10, 5))

SignalDetection.plot_roc(sdtList)
SignalDetection.fit_roc(sdtList)



class Metropolis:
    
    def __init__(self, logTarget, initialState):
        if not callable(logTarget):
            raise ValueError("logTarget must be a callable function")
        self.logTarget = logTarget
        if not isinstance(initialState, (int, float)):
            raise ValueError("initialState must be a numeric value")
        self.state = initialState
        self.acceptance_rate = 0.0
        self.step_size = 1.0
        self.samples = []  
    
    def __accept(self, proposal):
        log_alpha = self.logTarget(proposal) - self.logTarget(self.state)
        if np.log(np.random.rand()) < log_alpha:
            self.state = proposal
            return True
        else:
            return False
        
    def _propose(self):
        return np.random.normal(loc=self.state, scale=self.step_size)
        
    def adapt(self, blockLengths):
        if not isinstance(blockLengths, list):
            raise ValueError("blockLengths must be a list of integers")
        for i in range(len(blockLengths)):
            if not isinstance(blockLengths[i], int):
                raise ValueError("blockLengths must be a list of integers")        
        for i in range(len(blockLengths)):
            block_accepts = 0
            block_proposals = 0
            for j in range(blockLengths[i]):
                proposal = self._propose()
                if self.__accept(proposal):
                    block_accepts += 1
                block_proposals += 1
            acceptance_rate = block_accepts / block_proposals
            if acceptance_rate > 0.6:
                self.step_size *= 1.1
            elif acceptance_rate < 0.4:
                self.step_size /= 1.1
            self.acceptance_rate = 0.5*self.acceptance_rate + 0.5*acceptance_rate
        return self
    
    def sample(self, nSamples):
        if not isinstance(nSamples, int):
            raise ValueError("nSamples must be an integer")
        samples = [self.state]
        for i in range(nSamples):
            proposal = self._propose()
            self.__accept(proposal)
            samples.append(self.state)
        self.samples = samples
        return self
    
    def summary(self):
        if not hasattr(self, 'samples'):
            raise AttributeError("samples have not been generated yet")
        mean = np.mean(self.samples)
        ci025, ci975 = np.percentile(self.samples, [2.5, 97.5])
        summary_dict = {'mean': mean, 'c025': ci025, 'c975': ci975}
        return summary_dict
    
class TestSignalDetection(unittest.TestCase):
    def test_simulate(self):
        # Test with a single criterion value
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

        # Test with multiple criterion values
        dPrime       = 1.5
        criteriaList = [-0.5, 0, 0.5]
        signalCount  = 1000
        noiseCount   = 1000
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 3)
        for sdt in sdtList:
            self.assertLessEqual    (sdt.hits              ,  signalCount)
            self.assertLessEqual    (sdt.misses            ,  signalCount)
            self.assertLessEqual    (sdt.falseAlarms       ,  noiseCount)
            self.assertLessEqual    (sdt.correctRejections ,  noiseCount)

    def test_nLogLikelihood(self):
        sdt = SignalDetection(10, 5, 3, 12)
        hit_rate = 0.5
        false_alarm_rate = 0.2
        expected_nll = - (10 * np.log(hit_rate) +
                           5 * np.log(1-hit_rate) +
                           3 * np.log(false_alarm_rate) +
                          12 * np.log(1-false_alarm_rate))
        self.assertAlmostEqual(sdt.nLogLikelihood(hit_rate, false_alarm_rate),
                               expected_nll, places=6)

    def test_rocLoss(self):
        sdtList = [
            SignalDetection( 8, 2, 1, 9),
            SignalDetection(14, 1, 2, 8),
            SignalDetection(10, 3, 1, 9),
            SignalDetection(11, 2, 2, 8),
        ]
        a = 0
        expected = 99.3884206555698
        self.assertAlmostEqual(SignalDetection.rocLoss(a, sdtList), expected, places=4)

    def test_integration(self):
        dPrime = 1
        sdtList = SignalDetection.simulate(dPrime, [-1, 0, 1], 1e7, 1e7)
        aHat = SignalDetection.fit_roc(sdtList)
        self.assertAlmostEqual(aHat, dPrime, places=2)
        plt.close()
        


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

def fit_roc_bayesian(sdtList):

    # Define the log-likelihood function to optimize
    def loglik(a):
        return -SignalDetection.rocLoss(a, sdtList) + scipy.stats.norm.logpdf(a, loc = 0, scale = 10)

    # Create a Metropolis sampler object and adapt it to the target distribution
    sampler = Metropolis(logTarget = loglik, initialState = 0)
    sampler = sampler.adapt(blockLengths = [2000]*3)

    # Sample from the target distribution
    sampler = sampler.sample(nSamples = 4000)

    # Compute the summary statistics of the samples
    result  = sampler.summary()

    # Print the estimated value of the parameter a and its credible interval
    print(f"Estimated a: {result['mean']} ({result['c025']}, {result['c975']})")

    # Create a mosaic plot with four subplots
    fig, axes = plt.subplot_mosaic(
        [["ROC curve", "ROC curve", "traceplot"],
         ["ROC curve", "ROC curve", "histogram"]],
        constrained_layout = True
    )

    # Plot the ROC curve of the SDT data
    plt.sca(axes["ROC curve"])
    SignalDetection.plot_roc(sdtList = sdtList)

    # Compute the ROC curve for the estimated value of a and plot it
    xaxis = np.arange(start = 0.00,
                      stop  = 1.00,
                      step  = 0.01)

    plt.plot(xaxis, SignalDetection.rocCurve(xaxis, result['mean']), 'r-')

    # Shade the area between the lower and upper bounds of the credible interval
    plt.fill_between(x  = xaxis,
                     y1 = SignalDetection.rocCurve(xaxis, result['c025']),
                     y2 = SignalDetection.rocCurve(xaxis, result['c975']),
                     facecolor = 'r',
                     alpha     = 0.1)

    # Plot the trace of the sampler
    plt.sca(axes["traceplot"])
    plt.plot(sampler.samples)
    plt.xlabel('iteration')
    plt.ylabel('a')
    plt.title('Trace plot')

    # Plot the histogram of the samples
    plt.sca(axes["histogram"])
    plt.hist(sampler.samples,
             bins    = 51,
             density = True)
    plt.xlabel('a')
    plt.ylabel('density')
    plt.title('Histogram')

    # Show the plot
    plt.show()

# Define the number of SDT trials and generate a simulated dataset
sdtList = SignalDetection.simulate(dprime       = 1,
                                   criteriaList = [-1, 0, 1],
                                   signalCount  = 40,
                                   noiseCount   = 40)

# Fit the ROC curve to the simulated dataset
fit_roc_bayesian(sdtList)


# In[ ]:




