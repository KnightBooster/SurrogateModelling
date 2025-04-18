import numpy as np

def polynomial(dataX, weights):
    """Computes the polynomial function for the given x-coordinates and weights.

    Args:
        dataX (np.ndarray): One-dimensional array of x-coordinates.
        weights (np.ndarray): One-dimensional array of weights.

    Returns:
        np.ndarray: One-dimensional array of y-coordinates.
    """
    dataY = np.zeros(dataX.size)
    
    for row in range(dataY.size):
        for power, weight in enumerate(weights):
            dataY[row] += weight * (dataX[row] ** power)
            
    return dataY

def chebyshev(dataX, weights):
    """Computes the Chebyshev function for the given x-coordinates and weights.

    Args:
        dataX (np.ndarray): One-dimensional array of x-coordinates.
        weights (np.ndarray): One-dimensional array of weights.

    Returns:
        np.ndarray: One-dimensional array of y-coordinates.
    """
    dataY = np.zeros(dataX.size)
    
    for row in range(dataY.size):
        for weight in weights:
            if row == 0:
                dataY[row] += weight
            elif row == 1:
                dataY[row] += weight * dataX[row]
            else:
                dataY[row] += weight * (2 * dataX[row] * dataY[row - 1]) - (dataY[row - 2])

    return dataY

def sine(dataX, weights):
    """Computes the sine function for the given x-coordinates and weights.

    Args:
        dataX (np.ndarray): One-dimensional array of x-coordinates.
        weights (np.ndarray): One-dimensional array of weights.

    Returns:
        np.ndarray: One-dimensional array of y-coordinates.
    """
    dataY = np.zeros(dataX.size)
    
    for row in range(dataY.size):
        for frequencyFactor, weight in enumerate(weights, start=1):
            dataY[row] += weight * np.sin(frequencyFactor * np.pi * dataX[row])
            
    return dataY

def fitLinearModel(trainX: np.ndarray, trainY: np.ndarray, basis: str = "polynomial", degree: int = 2) -> callable:
    """Performs linear regression of a polynomial basis function of a given degree.

    Args:
        trainX (np.ndarray): One-dimensional array of all the x-coordinates of the training data points.
        trainY (np.ndarray): One-dimensional array of all the y-coordinates of the training data points.
        basis (str, optional):  The type of basis function to use for regression. Options are "polynomial", "chebyshev", and "sine". Defaults to "polynomial".
        degree (int, optional): The degree of the basis of the linear regression function. The degree must be a positive integer. Defaults to 2.
            - For polynomial basis, the degree is the degree of the polynomial function.
            - For chebyshev basis, the degree is the number of Chebyshev polynomials to use.    
            - For sine basis, the degree is the number of sine functions to use.

    Returns:
        callable: A function that takes a one-dimensional array of x-coordinates and returns the corresponding y-coordinates of the regression function.
    
    Examples:
        >>> degree = 2
        >>> basis = "polynomial"
        >>> trainX = np.array([0, 1, 2])
        >>> trainY = np.array([0, 1, 4])
        >>> dataX = np.array([0, 1, 2])
        >>> regressionFunction = linearRegression(trainX, trainY, basis=basis, degree=degree)
        >>> dataY = regressionFunction(dataX)
    """
        
    X = np.zeros((trainX.size, degree + 1))
    Y = np.zeros((trainY.size, 1))
        
    # Complete the Y matrix with the training data Y coordinates
    Y[:, 0] = trainY
    
    # Fill the X matrix with the basis functions evaluated at the training data X coordinates
    X[:, 0] = 1
    for row in range(X.shape[0]):
        for col in range(1, degree + 1):
            if basis == "polynomial":
                X[row, col] = trainX[row] ** col
            elif basis == "chebyshev":
                if col == 1:
                    X[row, col] = trainX[row]
                else:
                    X[row, col] = (2 * trainX[row] * X[row, col - 1]) - (X[row, col - 2])
            elif basis == "sine":
                X[row, col] = np.sin(col * np.pi * trainX[row])
            else:
                raise ValueError("Invalid basis type. Choose 'polynomial', 'chebyshev', or 'sine'.")
            
            # Set small values to zero to avoid numerical instability
            if abs(X[row, col]) < 0.000001:
                X[row, col] = 0

    # Solve the linear system to find the weights
    A = np.linalg.matmul(X.T, X)
    B = np.linalg.matmul(X.T, Y)
    W = np.linalg.solve(A, B).flatten()
    
    def regressionFunction(dataX: np.ndarray) -> np.ndarray:
        """

        Args:
            dataX (np.ndarray): One-dimensional array of x-coordinates to evaluate the regression function.

        Returns:
            np.ndarray: One-dimensional array of y-coordinates of _description_the regression function evaluated at the given x-coordinates.

        Examples:
            >>> dataX = np.array([0, 1, 2])
            >>> dataY = regressionFunction(dataX)
            >>> print(dataY)
        """
        if basis == "polynomial":
            dataY = polynomial(dataX, W)
        elif basis == "chebyshev":
            dataY = chebyshev(dataX, W)
        elif basis == "sine":
            dataY = sine(dataX, W)

        return dataY

    return regressionFunction