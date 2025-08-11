from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from utils import sigmoid

coef_func_signature = Callable[
    [np.ndarray, np.ndarray, np.random.Generator],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


def sample_random_uniform_coefficients(
    effective_dim_action_context: int,
    effective_dim_context: int,
    random_: np.random.Generator,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    context_coef_ = random_.uniform(-1, 1, size=effective_dim_context)
    action_coef_ = random_.uniform(-1, 1, size=effective_dim_action_context)
    context_action_coef_ = random_.uniform(
        -1, 1, size=(effective_dim_context, effective_dim_action_context)
    )
    return context_coef_, action_coef_, context_action_coef_


def _base_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    degree: int = 3,
    effective_dim_ratio: float = 1.0,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    z_score: bool = True,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Base function to define mean reward functions.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    degree: int, default=3
        Specifies the maximal degree of the polynomial feature transformations
        applied to both `context` and `action_context`.

    effective_dim_ratio: int, default=1.0
        Propotion of context dimensions relevant to the expected rewards.
        Specifically, after the polynomial feature transformation is applied to the original context vectors,
        only `dim_context * effective_dim_ratio` fraction of randomly selected dimensions
        will be used as relevant dimensions to generate expected rewards.

    z_score: boolean, default=True
        Boolean to enable/disable the calculation of a z-score over the resulting rewards. In case the environment
        is stationary, this can be turned on.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    poly = PolynomialFeatures(degree=degree)
    context_ = poly.fit_transform(context)
    action_context_ = poly.fit_transform(action_context)
    datasize, dim_context = context_.shape
    n_actions, dim_action_context = action_context_.shape
    random_ = np.random.default_rng(random_state)

    if effective_dim_ratio < 1.0:
        effective_dim_context = max(int(dim_context * effective_dim_ratio), 1)
        effective_dim_action_context = max(
            int(dim_action_context * effective_dim_ratio), 1
        )
        effective_context_ = context_[
            :, random_.choice(dim_context, effective_dim_context, replace=False)
        ]
        effective_action_context_ = action_context_[
            :,
            random_.choice(
                dim_action_context, effective_dim_action_context, replace=False
            ),
        ]
    else:
        effective_dim_context = dim_context
        effective_dim_action_context = dim_action_context
        effective_context_ = context_
        effective_action_context_ = action_context_

    context_coef_, action_coef_, context_action_coef_ = coef_function(
        n_rounds=datasize,
        effective_dim_action_context=effective_dim_action_context,
        effective_dim_context=effective_dim_context,
        random_=random_,
    )

    if context_coef_.shape[0] != datasize:
        # (n_rounds, n_actions)
        context_values = np.tile(effective_context_ @ context_coef_, (n_actions, 1)).T
    else:
        context_values = np.tile(
            np.sum(effective_context_ * context_coef_, axis=1), (n_actions, 1)
        ).T

    action_values = action_coef_ @ effective_action_context_.T
    if action_coef_.shape[0] != datasize:
        # (n_rounds, n_actions)
        action_values = np.tile(action_values, (datasize, 1))

    if action_coef_.shape[0] != datasize:
        # (n_rounds, dim_context) * (dim_context, dim_action_context) * (dim_action_context, n_actions) -> (n_rounds, n_actions)
        context_action_values = (
            effective_context_ @ context_action_coef_ @ effective_action_context_.T
        )
    else:
        effective_context_ = np.expand_dims(effective_context_, axis=1)
        context_action_coef_interactions = np.squeeze(
            np.matmul(effective_context_, context_action_coef_), axis=1
        )
        context_action_values = (
            context_action_coef_interactions @ effective_action_context_.T
        )

    expected_rewards = context_values + action_values + context_action_values
    if z_score:
        expected_rewards = (
            expected_rewards - expected_rewards.mean() / expected_rewards.std()
        )

    expected_rewards = degree * expected_rewards
    return expected_rewards


def logistic_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    z_score: bool = True,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for binary rewards.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    z_score: boolean, default=True
        Controls whether a z-score will be calculated over the computed logits.

    coef_function: Callable, default=sample_random_uniform_coefficients
        Function for generating the coefficients used for the context, action and context/action interactions.
        By default, the coefficients are randomly uniformly drawn.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    logits = _base_reward_function(
        context=context,
        action_context=action_context,
        degree=1,
        random_state=random_state,
        z_score=z_score,
        coef_function=coef_function,
    )

    return sigmoid(logits)
