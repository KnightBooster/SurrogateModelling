import numpy as np
import matplotlib.pyplot as plt

from src.sampling import randomSampling as rs, equiSpacedSampling as ess, latinHypercubeSampling as lhs
from src.regression import fitLinearModel
from src.regression import fitNonLinearModel
from src.regression import fitPCSI

def blackBoxFunction(x: float) -> float:
    return (6*x - 2)**2 * np.sin(12*x - 4)

numSamplesList = [5, 10, 15]
domain = [0, 1]
numSamples = numSamplesList[0]

randomSamples = rs(domain, numSamples)
equiSpacedSamples = ess(domain, numSamples)
latinHypercubeSamples = lhs(domain, numSamples)

# Generate the black-box function values
randomSamplesValues = blackBoxFunction(randomSamples)
equiSpacedSamplesValues = blackBoxFunction(equiSpacedSamples)
latinHypercubeSamplesValues = blackBoxFunction(latinHypercubeSamples)

basis = "polynomial"
samples = equiSpacedSamples
values = equiSpacedSamplesValues
degrees = [2, 3, 4]
# colors = plt.cm.jet(np.linspace(0, 1, len(degrees)))
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a', '#66a61e']


plt.figure(figsize=(10, 6))

x = np.linspace(domain[0], domain[1], 1000)
y = np.zeros((len(degrees), x.size))
for i, degree in enumerate(degrees):
    regressionFunction = fitLinearModel(samples, values, basis=basis, degree=degree)
    y[i] = regressionFunction(x)
    plt.plot(x, y[i], label=f'{basis.capitalize()} Degree {degree}', color=colors[i])
plt.scatter(samples, values, color='red', label='Random Samples')
plt.legend()
plt.title('Polynomial Basis Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(which='both')
plt.show()
