from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

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
