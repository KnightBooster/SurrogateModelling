import numpy as np
import matplotlib.pyplot as plt

from sampling import randomSampling as rs, equiSpacedSampling as ess, latinHypercubeSampling as lhs
from regressionModels import polynomialBasisRegression as lpb, chebyshevBasis as lcb, sineBasis as lsb
from regressionModels import nonLinearRegression as nlr
from regressionModels import piecewiseCubicSplineInterpolation as pcsi

def objectiveFunction(x: float) -> float:
    return (6*x - 2)**2 * np.sin(12*x - 4)

numSamplesList = [5, 10, 15]
domain = [0, 1]
numSamples = numSamplesList[2]

randomSamples = rs(domain, numSamples)
equiSpacedSamples = ess(domain, numSamples)
latinHypercubeSamples = lhs(domain, numSamples)

# Generate the black-box function values
randomSamplesValues = objectiveFunction(randomSamples)
equiSpacedSamplesValues = objectiveFunction(equiSpacedSamples)
latinHypercubeSamplesValues = objectiveFunction(latinHypercubeSamples)

x = np.linspace(domain[0], domain[1], 1000)
y = lpb(x, randomSamples, randomSamplesValues, 6)

plt.plot(x, y, label='Polynomial Basis Regression', color='blue')
plt.scatter(randomSamples, randomSamplesValues, color='red', label='Random Samples')
plt.legend()
plt.title('Polynomial Basis Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(which='both')
plt.show()
