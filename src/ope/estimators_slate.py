from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from utils import estimate_confidence_interval_by_bootstrap


@dataclass
class BaseSlateOffPolicyEstimator(metaclass=ABCMeta):
    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap."""
        raise NotImplementedError


@dataclass
class BaseSlateInverseProbabilityWeighting(BaseSlateOffPolicyEstimator):
    len_list: int

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each slate and slot (position).

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        behavior_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of behavior policy choosing a particular action at each position (slot) or
            joint probabilities of behavior policy choosing a whole slate/ranking of actions.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each slot/position or
            joint probabilities of evaluation policy choosing a whole slate/ranking of actions.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated for each slate and slot (position).

        """
        iw = evaluation_policy_pscore / behavior_policy_pscore
        estimated_rewards = reward * iw
        return estimated_rewards

    def _estimate_slate_confidence_interval_by_bootstrap(
        self,
        slate_id: np.ndarray,
        estimated_rewards: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated for each slate and slot (position).

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
        unique_slate = np.unique(slate_id)
        # sum estimated_rewards in each slate
        estimated_round_rewards = list()
        for slate in unique_slate:
            estimated_round_rewards.append(estimated_rewards[slate_id == slate].sum())
        estimated_round_rewards = np.array(estimated_round_rewards)
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateStandardIPS(BaseSlateInverseProbabilityWeighting):
    estimator_name: str = "sips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a slate action, i.e., :math:`\\pi_b(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                behavior_policy_pscore=pscore,
                evaluation_policy_pscore=evaluation_policy_pscore,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a slate action, i.e., :math:`\\pi_b(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

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
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateIndependentIPS(BaseSlateInverseProbabilityWeighting):
    estimator_name: str = "iips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of behavior policy choosing a particular action at each position (slot),
            i.e., :math:`\\pi_b(a_{i}(l) |x_i)`.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each position (slot),
            i.e., :math:`\\pi_e(a_{i}(l) |x_i)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                behavior_policy_pscore=pscore_item_position,
                evaluation_policy_pscore=evaluation_policy_pscore_item_position,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of behavior policy choosing a particular action at each position (slot),
            i.e., :math:`\\pi_b(a_{i}(l) |x_i)`.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each position (slot),
             i.e., :math:`\\pi_e(a_{i}(l) |x_i)`.

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
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_item_position,
            evaluation_policy_pscore=evaluation_policy_pscore_item_position,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateRewardInteractionIPS(BaseSlateInverseProbabilityWeighting):
    estimator_name: str = "rips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                behavior_policy_pscore=pscore_cascade,
                evaluation_policy_pscore=evaluation_policy_pscore_cascade,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

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
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_cascade,
            evaluation_policy_pscore=evaluation_policy_pscore_cascade,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateCascadeDoublyRobust(BaseSlateOffPolicyEstimator):
    len_list: int
    n_unique_action: int
    estimator_name: str = "cascade-dr"
