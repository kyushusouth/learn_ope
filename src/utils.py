from typing import Dict, Optional, Union

import numpy as np


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate sigmoid function."""
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def softmax(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate softmax function."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate confidence interval using bootstrap.

    Parameters
    ----------
    samples: array-like
        Empirical observed samples to be used to estimate cumulative distribution function.

    alpha: float, default=0.05
        Significance level.

    n_bootstrap_samples: int, default=10000
        Number of resampling performed in bootstrap sampling.

    random_state: int, default=None
        Controls the random seed in bootstrap sampling.

    Returns
    ----------
    estimated_confidence_interval: Dict[str, float]
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    boot_samples = list()
    random_ = np.random.default_rng(random_state)
    for _ in np.arange(n_bootstrap_samples):
        boot_samples.append(np.mean(random_.choice(samples, size=samples.shape[0])))
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))
    return {
        "mean": np.mean(boot_samples),
        f"{100 * (1.0 - alpha)}% CI (lower)": lower_bound,
        f"{100 * (1.0 - alpha)}% CI (upper)": upper_bound,
    }
