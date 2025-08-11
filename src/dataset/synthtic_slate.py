from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from dataset.base import BaseBanditDataset
from type_defs import BanditFeedback


@dataclass
class SyntheticSlateBanditDataset(BaseBanditDataset):
    seed: int
    dim_context: int
    behavior_policy_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    n_unique_action: int
    random_state: int

    def __post_init__(self):
        self._rand_gen = np.random.default_rng(self.seed)

        if self.behavior_policy_function is None:
            self.uniform_behavior_policy = (
                np.ones(self.n_unique_action) / self.n_unique_action
            )

        self.action_context = np.eye(self.n_unique_action, dtype=int)

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        return_pscore_item_position: bool,
        clip_logit_value: Optional[float],
    ) -> BanditFeedback:
        context = self._rand_gen.normal(0.0, 1.0, size=(n_rounds, self.dim_context))

        if self.behavior_policy_function is None:
            behavior_policy_logit_ = np.tile(
                self.uniform_behavior_policy, (n_rounds, 1)
            )
        else:
            behavior_policy_logit_ = self.behavior_policy_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )


def linear_behavior_policy_logit(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int],
    tau: Union[int, float],
) -> np.ndarray:
    """Linear contextual behavior policy for synthetic slate bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_unique_action, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    tau: int or float, default=1.0
        A temperature parameter to control the entropy of the behavior policy.
        As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

    Returns
    ---------
    logit value: array-like, shape (n_rounds, n_unique_action)
        Logit values to define the behavior policy.

    """
    rand_gen = np.random.default_rng(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    coef_ = rand_gen.uniform(0.0, 1.0, size=context.shape[1])
    action_coef_ = rand_gen.uniform(0.0, 1.0, size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_ + action_context[d] @ action_coef_

    return logits / tau
