from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ope.estimators_slate import BaseSlateOffPolicyEstimator
from type_defs import BanditFeedback


@dataclass
class SlateOffPolicyEvaluation:
    bandit_feedback: BanditFeedback
    ope_estimators: List[BaseSlateOffPolicyEstimator]

    def __post_init__(self) -> None:
        self.ope_estimators_ = dict()
        self.use_cascade_dr = False
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator
            # if isinstance(estimator, CascadeDR):
            #     self.use_cascade_dr = True

    def _create_estimator_inputs(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Create input dictionary to estimate policy value by subclasses of `BaseSlateOffPolicyEstimator`"""
        estimator_inputs = {
            input_: self.bandit_feedback[input_]
            for input_ in [
                "slate_id",
                "action",
                "reward",
                "position",
                "pscore",
                "pscore_item_position",
                "pscore_cascade",
            ]
            if input_ in self.bandit_feedback
        }
        estimator_inputs["evaluation_policy_pscore"] = evaluation_policy_pscore
        estimator_inputs["evaluation_policy_pscore_item_position"] = (
            evaluation_policy_pscore_item_position
        )
        estimator_inputs["evaluation_policy_pscore_cascade"] = (
            evaluation_policy_pscore_cascade
        )
        estimator_inputs["evaluation_policy_action_dist"] = (
            evaluation_policy_action_dist
        )
        estimator_inputs["q_hat"] = q_hat

        return estimator_inputs

    def estimate_policy_values(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each position (slot),
            i.e., :math:`\\pi_e(a_{i}(l) |x_i)`.

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_action_dist: array-like, shape (n_rounds * len_list * n_unique_action, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.
            Required when using `obp.ope.SlateCascadeDoublyRobust`.

        q_hat: array-like (n_rounds * len_list * n_unique_actions, )
            :math:`\\hat{Q}_l` for all unique actions,
            i.e., :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.
            Required when using `obp.ope.SlateCascadeDoublyRobust`.

        Returns
        ----------
        policy_value_dict: Dict[str, float]
            Dictionary containing the policy values estimated by OPE estimators.

        """
        policy_value_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
        )

        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_dict[estimator_name] = estimator.estimate_policy_value(
                **estimator_inputs
            )

        return policy_value_dict

    def estimate_intervals(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate the confidence intervals of the policy values using bootstrap.

        Parameters
        ------------
        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each position (slot),
             i.e., :math:`\\pi_e(a_{i}(l) |x_i)`.

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_action_dist: array-like, shape (n_rounds * len_list * n_unique_action, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        q_hat: array-like (n_rounds * len_list * n_unique_actions, )
            :math:`\\hat{Q}_l` for all unique actions,
            i.e., :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.
            Required when using `obp.ope.SlateCascadeDoublyRobust`.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        policy_value_interval_dict: Dict[str, Dict[str, float]]
            Dictionary containing confidence intervals of the estimated policy values.

        """
        policy_value_interval_dict = {}
        estimator_inputs = self._create_estimator_inputs(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
        )

        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_interval_dict[estimator_name] = estimator.estimate_interval(
                **estimator_inputs,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

        return policy_value_interval_dict

    def summarize_off_policy_estimates(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Summarize the estimated policy values and their confidence intervals estimated by bootstrap.

        Parameters
        ------------
        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each position (slot),
             i.e., :math:`\\pi_e(a_{i}(l) |x_i)`.

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.


        evaluation_policy_action_dist: array-like, shape (n_rounds * len_list * n_unique_action, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        q_hat: array-like (n_rounds * len_list * n_unique_actions, )
            :math:`\\hat{Q}_l` for all unique actions,
            i.e., :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.
            Required when using `obp.ope.SlateCascadeDoublyRobust`.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        (policy_value_df, policy_value_interval_df): Tuple[DataFrame, DataFrame]
            Policy values and their confidence intervals estimated by OPE estimators.

        """
        policy_value_df = pd.DataFrame(
            self.estimate_policy_values(
                evaluation_policy_pscore=evaluation_policy_pscore,
                evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
                evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
                q_hat=q_hat,
            ),
            index=["estimated_policy_value"],
        )
        policy_value_interval_df = pd.DataFrame(
            self.estimate_intervals(
                evaluation_policy_pscore=evaluation_policy_pscore,
                evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
                evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
                q_hat=q_hat,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )
        )
        policy_value_df = policy_value_df.T
        return policy_value_df, policy_value_interval_df.T

    def evaluate_performance_of_estimators(
        self,
        ground_truth_policy_value: float,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        metric: str = "se",
    ) -> Dict[str, float]:
        """Evaluate the accuracy of OPE estimators.

        Parameters
        ----------
        ground_truth_policy_value: float
            Ground_truth policy value of evaluation policy, i.e., :math:`V(\\pi)`.
            With Open Bandit Dataset, we use an on-policy estimate of the policy value as its ground-truth.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each position (slot),
            i.e., :math:`\\pi_e(a_{i}(l) |x_i)`.

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.


        evaluation_policy_action_dist: array-like, shape (n_rounds * len_list * n_unique_action, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        q_hat: array-like (n_rounds * len_list * n_unique_actions, )
            :math:`\\hat{Q}_l` for all unique actions,
            i.e., :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.
            Required when using `obp.ope.SlateCascadeDoublyRobust`.

        metric: str, default="se"
            Evaluation metric used to evaluate and compare the estimation performance of OPE estimators.
            Must be either "relative-ee" or "se".

        Returns
        ----------
        eval_metric_ope_dict: Dict[str, float]
            Dictionary containing the value of evaluation metric for the estimation performance of OPE estimators.

        """
        eval_metric_ope_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_policy_value = estimator.estimate_policy_value(**estimator_inputs)
            if metric == "relative-ee":
                relative_ee_ = estimated_policy_value - ground_truth_policy_value
                relative_ee_ /= ground_truth_policy_value
                eval_metric_ope_dict[estimator_name] = np.abs(relative_ee_)
            elif metric == "se":
                se_ = (estimated_policy_value - ground_truth_policy_value) ** 2
                eval_metric_ope_dict[estimator_name] = se_
        return eval_metric_ope_dict

    def summarize_estimators_comparison(
        self,
        ground_truth_policy_value: float,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        metric: str = "se",
    ) -> pd.DataFrame:
        """Summarize the performance comparison among OPE estimators.

        Parameters
        ----------
        ground_truth_policy_value: float
            Ground_truth policy value of evaluation policy, i.e., :math:`V(\\pi)`.
            With Open Bandit Dataset, we use an on-policy estimate of the policy value as ground-truth.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each position (slot),
            i.e., :math:`\\pi_e(a_{i}(l) |x_i)`.

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_action_dist: array-like, shape (n_rounds * len_list * n_unique_action, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        q_hat: array-like (n_rounds * len_list * n_unique_actions, )
            :math:`\\hat{Q}_l` for all unique actions,
            i.e., :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.
            Required when using `obp.ope.SlateCascadeDoublyRobust`.

        metric: str, default="se"
            Evaluation metric used to evaluate and compare the estimation performance of OPE estimators.
            Must be either "relative-ee" or "se".

        Returns
        ----------
        eval_metric_ope_df: DataFrame
            Results of performance comparison among OPE estimators.

        """
        eval_metric_ope_df = pd.DataFrame(
            self.evaluate_performance_of_estimators(
                ground_truth_policy_value=ground_truth_policy_value,
                evaluation_policy_pscore=evaluation_policy_pscore,
                evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
                evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
                q_hat=q_hat,
                metric=metric,
            ),
            index=[metric],
        )
        return eval_metric_ope_df.T
