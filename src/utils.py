from typing import Union

import numpy as np


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate sigmoid function."""
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))
