from .linear_regression import fitLinearModel
from .non_linear_regression import fitNonLinearModel
from .piecewise_cubic_spline_interpolation import fitPCSI
from .regression_metrics import (
    computeMeanSquareError,
    computeRsquared
)

__author__ = "Abhishek Mukherjee"
__email__ = "abhishekmukherjee.iist@gmail.com"
__version__ = "0.1.0"
__status__ = "Prototype"
__description__ = "Regression models and interpolation methods for data analysis."
__url__ = ""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 Abhishek Mukherjee"
__date__ = "2025-04-14"
__maintainer__ = "Abhishek Mukherjee"
__maintainer_email__ = "abhishekmukherjee.iist@gmail.com"

__all__ = [
    "linear_regression",
    "non_linear_regression",
    "piecewise_cubic_spline_interpolation",
    "computeMeanSquareError",
    "computeRsquared"
]

# print(f"Regression module {__version__} loaded successfully.")
print(f"Copyright (c) 2025 Abhishek Mukherjee. All rights reserved.")
print(f"Module: {__name__}")
print(f"Version: {__version__}")
print(f"Status: {__status__}")
