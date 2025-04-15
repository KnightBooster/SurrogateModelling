import numpy as np

def polynomialBasisRegression(dataX, trainX: np.ndarray, trainY: np.ndarray, degree: int = 2) -> np.ndarray:   
    """Performs linear regression of a polynomial basis function of a given degree.

    Args:
        dataX (np.ndarray): One-dimensional array of all the x-coordinates of the regression data points.
        trainX (np.ndarray): One-dimensional array of all the x-coordinates of the training data points.
        trainY (np.ndarray): One-dimensional array of all the y-coordinates of the training data points.
        degree (int): The degree of the polynomial basis linear regression function. The degree must be a positive integer. Defaults to 2.

    Returns:
        np.ndarray: The y-coordinates of the regression data points evaluated at the given x-coordinates. 
    
    Examples:
        >>> x = np.array([1, 2, 3])
        >>> y = blackBoxFunction(x)
        >>> degree = 2
        >>> coefficients = polynomialBasisRegression(x, y, degree)
    """
        
    X = np.zeros((trainX.size, degree + 1))
    Y = np.zeros((trainY.size, 1))
    
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            X[row, col] = trainX[row] ** col
            
    for row in range(Y.shape[0]):
        Y[row, 0] = trainY[row]

    W = np.linalg.matmul(np.linalg.inv(np.linalg.matmul(X.T, X)), np.linalg.matmul(X.T, Y)).flatten()
    
    dataY = np.zeros(dataX.size)
    
    for row in range(dataY.size):
        for col in range(W.size):
            dataY[row] += W[col] * (dataX[row] ** col)
            
    return dataY
        
def chebyshevBasis(order):
    pass

def sineBasis(frequency):
    pass

def nonLinearRegression():
    pass

def piecewiseCubicSplineInterpolation():
    pass

