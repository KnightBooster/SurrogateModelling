import numpy as np

def computeRsquared(trainX: np.ndarray, trainY: np.ndarray, regressionFunction: callable) -> float:
    """Computes the R-squared value of the regression function.

    Args:
        trainX (np.ndarray): One-dimensional array of all the x-coordinates of the training data points.
        trainY (np.ndarray): One-dimensional array of all the y-coordinates of the training data points.
        regressionFunction (callable): The regression function to evaluate.

    Returns:
        float: The R-squared value of the regression function.
    """
    dataY = regressionFunction(trainX)
    ss_res = np.sum((trainY - dataY) ** 2)
    ss_tot = np.sum((trainY - np.mean(trainY)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared

def computeMeanSquareError(trainX: np.ndarray, trainY: np.ndarray, regressionFunction: callable, shouldReturnRMSE: bool = False) -> float:
    """Computes the mean square error of the regression function.

    Args:
        trainX (np.ndarray): One-dimensional array of all the x-coordinates of the training data points.
        trainY (np.ndarray): One-dimensional array of all the y-coordinates of the training data points.
        regressionFunction (callable): The regression function to evaluate.

    Returns:
        float: The mean square error of the regression function.
    """
    dataY = regressionFunction(trainX)
    mse = np.mean((trainY - dataY) ** 2)
    rmse = np.sqrt(mse)
    
    if shouldReturnRMSE:
        return mse, rmse
    else:
        return mse
