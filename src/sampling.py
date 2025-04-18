import numpy as np
from pyDOE import lhs

def randomSampling(domain, numSamples):
    return np.random.uniform(domain[0], domain[1], numSamples)

def equiSpacedSampling(domain, numSamples):
    return np.linspace(domain[0], domain[1], numSamples, endpoint=True)

def latinHypercubeSampling(domain, numSamples):
    return (lhs(1, samples=numSamples) * (domain[1] - domain[0])) + domain[0]