from dataclasses import dataclass
from itertools import permutations, product
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.special import logit
from scipy.stats import truncnorm
from tqdm import tqdm

from dataset.base import BaseBanditDataset
from type_defs import BanditFeedback
from utils import sigmoid, softmax


@dataclass
class SyntheticSlateBanditDataset(BaseBanditDataset):
    dim_context: int
    behavior_policy_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    n_unique_action: int
    random_state: int
    len_list: int
    is_factorizable: bool
    base_reward_function: Optional[
        Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ]
    ]
    reward_type: str
    reward_structure: str
    click_model: str
    eta: float = 1.0

    def __post_init__(self):
        self.random_ = np.random.default_rng(self.random_state)

        if self.behavior_policy_function is None:
            self.uniform_behavior_policy = (
                np.ones(self.n_unique_action) / self.n_unique_action
            )

        self.action_context = np.eye(self.n_unique_action, dtype=int)
        if self.base_reward_function is not None:
            self.reward_function = action_interaction_reward_function

        if self.click_model == "pbm":
            self.attractiveness = np.ones(self.len_list, dtype=float)
            self.exam_weight = (1.0 / np.arange(1, self.len_list + 1)) ** self.eta
        elif self.click_model == "cascade":
            self.attractiveness = (1.0 / np.arange(1, self.len_list + 1)) ** self.eta
            self.exam_weight = np.ones(self.len_list, dtype=float)
        else:
            self.attractiveness = np.ones(self.len_list, dtype=float)
            self.exam_weight = np.ones(self.len_list, dtype=float)

        if self.reward_structure in ["cascade_additive", "standard_additive"]:
            # generate additive action interaction weight matrix of (n_unique_action, n_unique_action)
            self.action_interaction_weight_matrix = generate_symmetric_matrix(
                n_unique_action=self.n_unique_action, random_state=self.random_state
            )

    def sample_action_and_obtain_pscore(
        self,
        behavior_policy_logit_: np.ndarray,
        n_rounds: int,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Sample action and obtain the three variants of the propensity scores.

        Parameters
        ------------
        behavior_policy_logit_: array-like, shape (n_rounds, n_actions)
            Logit values given context (:math:`x`).

        n_rounds: int
            Data size of synthetic logged data.

        return_pscore_item_position: bool, default=True
            Whether to compute `pscore_item_position` and include it in the logged data.
            When `n_actions` and `len_list` are large, `return_pscore_item_position`=True can lead to a long computation time.

        clip_logit_value: Optional[float], default=None
            A float parameter used to clip logit values (<= `700.`).
            When None, clipping is not applied to softmax values when obtaining `pscore_item_position`.
            When a float value is given, logit values are clipped when calculating softmax values.
            When `n_actions` and `len_list` are large, `clip_logit_value`=None can lead to a long computation time.

        Returns
        ----------
        action: array-like, shape (n_rounds * len_list)
            Actions sampled by the behavior policy.
            Actions sampled within slate `i` is stored in `action[`i` * `len_list`: (`i + 1`) * `len_list`]`.

        pscore: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the slate actions given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,1}, a_{i,2}, \\ldots, a_{i,L} | x_{i} )`.

        pscore_item_position: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the action of the :math:`l`-th slot given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,l} | x_{i} )`.

        pscore_cascade: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the actions of the top :math:`l` slots given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,1}, a_{i,2}, \\ldots, a_{i,l} | x_{i} )`.

        """
        action = np.zeros(n_rounds * self.len_list, dtype=int)
        pscore_cascade = np.zeros(n_rounds * self.len_list)
        pscore = np.zeros(n_rounds * self.len_list)

        if return_pscore_item_position:
            pscore_item_position = np.zeros(n_rounds * self.len_list)
            if not self.is_factorizable and self.behavior_policy_function is not None:
                enumerated_slate_actions = [
                    _
                    for _ in permutations(
                        np.arange(self.n_unique_action), self.len_list
                    )
                ]
                enumerated_slate_actions = np.array(
                    enumerated_slate_actions
                )  # (n_comb, len_list)
        else:
            pscore_item_position = None

        if return_pscore_item_position and clip_logit_value is not None:
            behavior_policy_softmax_ = np.exp(
                np.minimum(behavior_policy_logit_, clip_logit_value)
            )

        for i in tqdm(
            np.arange(n_rounds),
            desc="[sample_action_and_obtain_pscore]",
            total=n_rounds,
        ):
            unique_action_set = np.arange(self.n_unique_action)
            score_ = softmax(behavior_policy_logit_[i : i + 1, unique_action_set])[
                0
            ]  # (n_unique_action,)
            pscore_i = 1.0

            for pos_ in np.arange(self.len_list):
                sampled_action = self.random_.choice(
                    unique_action_set, p=score_, replace=False
                )
                action[i * self.len_list + pos_] = sampled_action
                sampled_action_index = np.where(unique_action_set == sampled_action)[0][
                    0
                ]

                pscore_i *= score_[sampled_action_index]
                pscore_cascade[i * self.len_list + pos_] = pscore_i

                if not self.is_factorizable and pos_ != self.len_list - 1:
                    unique_action_set = np.delete(
                        unique_action_set, unique_action_set == sampled_action
                    )
                    score_ = softmax(
                        behavior_policy_logit_[i : i + 1, unique_action_set]
                    )[0]

                if return_pscore_item_position:
                    if self.behavior_policy_function is None:  # uniform random
                        pscore_item_pos_i_l = 1 / self.n_unique_action
                    elif self.is_factorizable:
                        pscore_item_pos_i_l = score_[sampled_action_index]
                    elif pos_ == 0:
                        pscore_item_pos_i_l = pscore_i
                    else:
                        if isinstance(clip_logit_value, float):
                            pscores = self._calc_pscore_given_policy_softmax(
                                all_slate_actions=enumerated_slate_actions,
                                policy_softmax_i_=behavior_policy_softmax_[i],
                            )
                        else:
                            pscores = self._calc_pscore_given_policy_logit(
                                all_slate_actions=enumerated_slate_actions,
                                policy_logit_i_=behavior_policy_logit_[i],
                            )
                        pscore_item_pos_i_l = pscores[
                            enumerated_slate_actions[:, pos_] == sampled_action
                        ].sum()
                    pscore_item_position[i * self.len_list + pos_] = pscore_item_pos_i_l

            start_idx = i * self.len_list
            end_idx = start_idx + self.len_list
            pscore[start_idx:end_idx] = pscore_i

        return action, pscore_cascade, pscore, pscore_item_position

    def _calc_pscore_given_policy_logit(
        self, all_slate_actions: np.ndarray, policy_logit_i_: np.ndarray
    ) -> np.ndarray:
        """Calculate the propensity score of all possible slate actions given a particular policy_logit.

        Parameters
        ------------
        all_slate_actions: array-like, (n_action, len_list)
            All possible slate actions.

        policy_logit_i_: array-like, (n_unique_action, )
            Logit values given context (:math:`x`), which defines the distribution over actions of the policy.

        Returns
        ------------
        pscores: array-like, (n_action, )
            Propensity scores of all slate actions.

        """
        n_actions = len(all_slate_actions)
        unique_action_set_2d = np.tile(np.arange(self.n_unique_action), (n_actions, 1))
        pscores = np.ones(n_actions)

        for pos_ in np.arange(self.len_list):
            action_index = np.where(
                unique_action_set_2d == all_slate_actions[:, pos_][:, np.newaxis]
            )[1]
            pscores *= softmax(policy_logit_i_[unique_action_set_2d])[
                np.arange(n_actions), action_index
            ]

            if pos_ + 1 != self.len_list:
                # 今選んだアクションを次選べないようにマスクする
                mask = np.ones((n_actions, self.n_unique_action - pos_))
                mask[np.arange(n_actions), action_index] = 0
                unique_action_set_2d = unique_action_set_2d[mask.astype(bool)].reshape(
                    (-1, self.n_unique_action - pos_ - 1)
                )

        return pscores

    def _calc_pscore_given_policy_softmax(
        self, all_slate_actions: np.ndarray, policy_softmax_i_: np.ndarray
    ) -> np.ndarray:
        """Calculate the propensity score of all possible slate actions given a particular policy_softmax.

        Parameters
        ------------
        all_slate_actions: array-like, (n_action, len_list)
            All possible slate actions.

        policy_softmax_i_: array-like, (n_unique_action, )
            Policy softmax values given context (:math:`x`).

        Returns
        ------------
        pscores: array-like, (n_action, )
            Propensity scores of all slate actions.

        """
        n_actions = len(all_slate_actions)
        unique_action_set_2d = np.tile(
            np.arange(self.n_unique_action), (n_actions, 1)
        )  # (n_actions, n_unique_actions)
        pscores = np.ones(n_actions)

        for pos_ in np.arange(self.len_list):
            action_index = np.where(
                unique_action_set_2d == all_slate_actions[:, pos_][:, np.newaxis]
            )[1]  # (n_actions,)
            score_ = policy_softmax_i_[unique_action_set_2d]
            pscores *= np.divide(score_, score_.sum(axis=1, keepdims=True))[
                np.arange(n_actions), action_index
            ]

            if pos_ + 1 != self.len_list:
                mask = np.ones((n_actions, self.n_unique_action - pos_))
                mask[np.arange(n_actions), action_index] = 0
                unique_action_set_2d = unique_action_set_2d[mask.astype(bool)].reshape(
                    (-1, self.n_unique_action - pos_ - 1)
                )

        return pscores

    def sample_contextfree_expected_reward(
        self, random_state: Optional[int] = None
    ) -> np.ndarray:
        """Define context independent expected rewards for each action and slot.

        Parameters
        -----------
        random_state: int, default=None
            Controls the random seed in sampling dataset.

        """
        random_ = np.random.default_rng(random_state)
        return random_.uniform(size=(self.n_unique_action, self.len_list))

    def sample_reward_given_expected_reward(
        self, expected_reward_factual: np.ndarray
    ) -> np.ndarray:
        """Sample reward variables given actions observed at each slot.

        Parameters
        ------------
        expected_reward_factual: array-like, shape (n_rounds, len_list)
            Expected rewards given observed actions and contexts.

        Returns
        ----------
        reward: array-like, shape (n_rounds, len_list)
            Sampled rewards.

        """
        expected_reward_factual *= self.exam_weight

        if self.reward_type == "binary":
            sampled_reward_list = []
            discount_factors = np.ones(expected_reward_factual.shape[0])
            sampled_rewards_at_position = np.zeros(expected_reward_factual.shape[0])
            for pos_ in np.arange(self.len_list):
                discount_factors *= sampled_rewards_at_position * self.attractiveness[
                    pos_
                ] + (1 - sampled_rewards_at_position)
                expected_reward_factual_at_position = (
                    discount_factors * expected_reward_factual[:, pos_]
                )
                sampled_rewards_at_position = self.random_.binomial(
                    n=1, p=expected_reward_factual_at_position
                )
                sampled_reward_list.append(sampled_rewards_at_position)
            reward = np.array(sampled_reward_list).T
        elif self.reward_type == "continuous":
            reward = np.zeros(expected_reward_factual.shape)
            for pos_ in np.arange(self.len_list):
                mean = expected_reward_factual[:, pos_]
                a = (self.reward_min - mean) / self.reward_std
                b = (self.reward_max - mean) / self.reward_std
                reward[:, pos_] = truncnorm.rvs(
                    a=a,
                    b=b,
                    loc=mean,
                    scale=self.reward_std,
                    random_state=self.random_state,
                )
        else:
            raise NotImplementedError

        return reward

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        return_pscore_item_position: bool,
        clip_logit_value: Optional[float] = None,
    ) -> BanditFeedback:
        context = self.random_.normal(0.0, 1.0, size=(n_rounds, self.dim_context))

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

        (
            action,
            pscore_cascade,
            pscore,
            pscore_item_position,
        ) = self.sample_action_and_obtain_pscore(
            behavior_policy_logit_=behavior_policy_logit_,
            n_rounds=n_rounds,
            return_pscore_item_position=return_pscore_item_position,
            clip_logit_value=clip_logit_value,
        )

        if self.base_reward_function is None:
            expected_reward = self.sample_contextfree_expected_reward(
                random_state=self.random_state
            )  # (n_unique_action, len_list)
            expected_reward_tile = np.tile(
                expected_reward[None, :, :], (n_rounds, 1, 1)
            )  # (n_rounds, n_unique_action, len_list)
            action_2d = action.reshape((n_rounds, self.len_list))
            expected_reward_factual = np.array(
                [
                    expected_reward_tile[:, action_2d[:, pos_], pos_]
                    for pos_ in np.arange(self.len_list)
                ]
            ).T
        else:
            expected_reward_factual = self.reward_function(
                context=context,
                action_context=self.action_context,
                action=action,
                action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                base_reward_function=self.base_reward_function,
                reward_type=self.reward_type,
                reward_structure=self.reward_structure,
                len_list=self.len_list,
                random_state=self.random_state,
            )

        reward = self.sample_reward_given_expected_reward(
            expected_reward_factual=expected_reward_factual
        )

        return dict(
            n_rounds=n_rounds,
            n_unique_action=self.n_unique_action,
            slate_id=np.repeat(np.arange(n_rounds), self.len_list),
            context=context,
            action_context=self.action_context,
            action=action,
            position=np.tile(np.arange(self.len_list), n_rounds),
            reward=reward.reshape(action.shape[0]),
            expected_reward_factual=expected_reward_factual.reshape(action.shape[0]),
            pscore_cascade=pscore_cascade,
            pscore=pscore,
            pscore_item_position=pscore_item_position,
        )

    def calc_on_policy_policy_value(
        self, reward: np.ndarray, slate_id: np.ndarray
    ) -> float:
        """Calculate the policy value of given reward and slate_id.

        Parameters
        -----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate index.

        Returns
        ----------
        policy_value: float
            The on-policy policy value estimate of the behavior policy.

        """
        return reward.sum() / np.unique(slate_id).shape[0]

    def obtain_pscore_given_evaluation_policy_logit(
        self,
        action: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ):
        """Calculate the propensity score given particular logit values to define the evaluation policy.

        Parameters
        ------------
        action: array-like, (n_rounds * len_list, )
            Action chosen by the behavior policy.

        evaluation_policy_logit_: array-like, (n_rounds, n_unique_action)
            Logit values to define the evaluation policy.

        return_pscore_item_position: bool, default=True
            Whether to compute `pscore_item_position` and include it in the logged data.
            When `n_actions` and `len_list` are large, `return_pscore_item_position`=True can lead to a long computation time.

        clip_logit_value: Optional[float], default=None
            A float parameter used to clip logit values (<= `700.`).
            When None, clipping is not applied to softmax values when obtaining `pscore_item_position`.
            When a float value is given, logit values are clipped when calculating softmax values.
            When `n_actions` and `len_list` are large, `clip_logit_value`=None can lead to a long computation time.

        """
        n_rounds = action.reshape((-1, self.len_list)).shape[0]
        pscore_cascade = np.zeros(n_rounds * self.len_list)
        pscore = np.zeros(n_rounds * self.len_list)

        if return_pscore_item_position:
            pscore_item_position = np.zeros(n_rounds * self.len_list)
            if not self.is_factorizable:
                enumerated_slate_actions = [
                    _
                    for _ in permutations(
                        np.arange(self.n_unique_action), self.len_list
                    )
                ]
                enumerated_slate_actions = np.array(enumerated_slate_actions)
        else:
            pscore_item_position = None

        if return_pscore_item_position and clip_logit_value is not None:
            evaluation_policy_softmax_ = np.exp(
                np.minimum(evaluation_policy_logit_, clip_logit_value)
            )

        for i in tqdm(
            np.arange(n_rounds),
            desc="[obtain_pscore_given_evaluation_policy_logit]",
            total=n_rounds,
        ):
            unique_action_set = np.arange(self.n_unique_action)
            score_ = softmax(evaluation_policy_logit_[i : i + 1])[0]
            pscore_i = 1.0

            for pos_ in np.arange(self.len_list):
                action_ = action[i * self.len_list + pos_]
                action_index_ = np.where(unique_action_set == action_)[0][0]
                pscore_i *= score_[action_index_]
                pscore_cascade[i * self.len_list + pos_] = pscore_i

                if not self.is_factorizable and pos_ != self.len_list - 1:
                    unique_action_set = np.delete(
                        unique_action_set, unique_action_set == action_
                    )
                    score_ = softmax(
                        evaluation_policy_logit_[i : i + 1, unique_action_set]
                    )[0]

                if return_pscore_item_position:
                    if pos_ == 0:
                        pscore_item_pos_i_l = pscore_i
                    elif self.is_factorizable:
                        pscore_item_pos_i_l = score_[action_index_]
                    else:
                        if isinstance(clip_logit_value, float):
                            pscores = self._calc_pscore_given_policy_softmax(
                                all_slate_actions=enumerated_slate_actions,
                                policy_softmax_i_=evaluation_policy_softmax_[i],
                            )
                        else:
                            pscores = self._calc_pscore_given_policy_logit(
                                all_slate_actions=enumerated_slate_actions,
                                policy_logit_i_=evaluation_policy_logit_[i],
                            )
                        pscore_item_pos_i_l = pscores[
                            enumerated_slate_actions[:, pos_] == action_
                        ].sum()
                    pscore_item_position[i * self.len_list + pos_] = pscore_item_pos_i_l

            start_idx = i * self.len_list
            end_idx = start_idx + self.len_list
            pscore[start_idx:end_idx] = pscore_i

        return pscore, pscore_item_position, pscore_cascade

    def calc_ground_truth_policy_value(
        self,
        context: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
    ):
        """Calculate the ground-truth policy value of given evaluation policy logit and contexts.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        evaluation_policy_logit_: array-like, shape (n_rounds, n_unique_action)
            Logit values to define the evaluation policy.

        """
        if self.is_factorizable:
            enumerated_slate_actions = [
                _
                for _ in product(np.arange(self.n_unique_action), repeat=self.len_list)
            ]
        else:
            enumerated_slate_actions = [
                _ for _ in permutations(np.arange(self.n_unique_action), self.len_list)
            ]
        enumerated_slate_actions = np.array(enumerated_slate_actions).astype("int8")
        n_slate_actions = len(enumerated_slate_actions)
        n_rounds = len(evaluation_policy_logit_)

        pscores = []
        n_enumerated_slate_actions = len(enumerated_slate_actions)
        if self.is_factorizable:
            for action_list in tqdm(
                enumerated_slate_actions,
                desc="[calc_ground_truth_policy_value (pscore)]",
                total=n_enumerated_slate_actions,
            ):
                pscores.append(
                    softmax(evaluation_policy_logit_)[:, action_list].prod(1)
                )
            pscores = np.array(pscores).T
        else:
            for i in tqdm(
                np.arange(n_rounds),
                desc="[calc_ground_truth_policy_value (pscore)]",
                total=n_rounds,
            ):
                pscores.append(
                    self._calc_pscore_given_policy_logit(
                        all_slate_actions=enumerated_slate_actions,
                        policy_logit_i_=evaluation_policy_logit_[i],
                    )
                )
            pscores = np.array(pscores)

        # calculate expected slate-level reward for each combinatorial set of items (i.e., slate actions)
        if self.base_reward_function is None:
            expected_slot_reward = self.sample_contextfree_expected_reward(
                random_state=self.random_state
            )
            expected_slot_reward_tile = np.tile(
                expected_slot_reward, (n_rounds * n_slate_actions, 1, 1)
            )
            expected_slate_rewards = np.array(
                [
                    expected_slot_reward_tile[
                        np.arange(n_slate_actions) % n_slate_actions,
                        np.array(enumerated_slate_actions)[:, pos_],
                        pos_,
                    ]
                    for pos_ in np.arange(self.len_list)
                ]
            ).T
            policy_value = (pscores * expected_slate_rewards.sum(axis=1)).sum()
        else:
            n_batch = (
                n_rounds * n_enumerated_slate_actions * self.len_list - 1
            ) // 10**7 + 1
            batch_size = (n_rounds - 1) // n_batch + 1
            n_batch = (n_rounds - 1) // batch_size + 1

            policy_value = 0.0
            for batch_idx in tqdm(
                np.arange(n_batch),
                desc=f"[calc_ground_truth_policy_value (expected reward), batch_size={batch_size}]",
                total=n_batch,
            ):
                context_ = context[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]
                pscores_ = pscores[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]

                expected_slate_rewards_ = self.reward_function(
                    context=context_,
                    action_context=self.action_context,
                    action=enumerated_slate_actions.flatten(),
                    action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                    base_reward_function=self.base_reward_function,
                    reward_type=self.reward_type,
                    reward_structure=self.reward_structure,
                    len_list=self.len_list,
                    is_enumerated=True,
                    random_state=self.random_state,
                )

                # click models based on expected reward
                expected_slate_rewards_ *= self.exam_weight
                if self.reward_type == "binary":
                    discount_factors = np.ones(expected_slate_rewards_.shape[0])
                    previous_slot_expected_reward = np.zeros(
                        expected_slate_rewards_.shape[0]
                    )
                    for pos_ in np.arange(self.len_list):
                        discount_factors *= (
                            previous_slot_expected_reward * self.attractiveness[pos_]
                            + (1 - previous_slot_expected_reward)
                        )
                        expected_slate_rewards_[:, pos_] = (
                            discount_factors * expected_slate_rewards_[:, pos_]
                        )
                        previous_slot_expected_reward = expected_slate_rewards_[:, pos_]

                policy_value += (
                    pscores_.flatten() * expected_slate_rewards_.sum(axis=1)
                ).sum()
            policy_value /= n_rounds

        return policy_value


def linear_behavior_policy_logit(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int],
    tau: Union[int, float] = 1.0,
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


def generate_symmetric_matrix(n_unique_action: int, random_state: int) -> np.ndarray:
    """Generate symmetric matrix

    Parameters
    -----------
    n_unique_action: int (>= len_list)
        Number of unique actions.

    random_state: int
        Controls the random seed in sampling elements of matrix.

    Returns
    ---------
    symmetric_matrix: array-like, shape (n_unique_action, n_unique_action)

    """
    random_ = np.random.default_rng(random_state)
    base_matrix = random_.normal(0, 5, size=(n_unique_action, n_unique_action))
    symmetric_matrix = (
        np.tril(base_matrix) + np.tril(base_matrix).T - np.diag(base_matrix.diagonal())
    )
    return symmetric_matrix


def action_interaction_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    action: np.ndarray,
    base_reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    reward_type: str,
    reward_structure: str,
    action_interaction_weight_matrix: np.ndarray,
    len_list: int,
    is_enumerated: bool = False,
    random_state: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Reward function incorporating interactions among combinatorial action

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_unique_action, dim_action_context)
        Vector representation of actions.

    action: array-like, shape (n_rounds * len_list, ) or (len(enumerated_slate_actions) * len_list, )
        When `is_enumerated`=False, action corresponds to actions sampled by a (often behavior) policy.
        In this case, actions sampled within slate `i` is stored in `action[`i` * `len_list`: (`i + 1`) * `len_list`]`.
        When `is_enumerated`=True, action corresponds to the enumerated all possible combinatorial actions.

    base_reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function to define the expected reward, i.e., :math:`q: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary',the expected rewards are transformed by logit function.

    reward_structure: str
        Reward structure.
        Must be one of 'standard_additive', 'cascade_additive', 'standard_decay', or 'cascade_decay'.

    action_interaction_weight_matrix (`W`): array-like, shape (n_unique_action, n_unique_action) or (len_list, len_list)
        When using an additive-type reward_structure, `W(i, j)` defines the interaction between action `i` and `j`.
        When using an decay-type reward_structure, `W(i, j)` defines the weight of how the expected reward of slot `i` affects that of slot `j`.
        See the experiment section of Kiyohara et al.(2022) for details.

    len_list: int (> 1)
        Length of a list/ranking of actions, slate size.

    is_enumerate: bool
        Whether `action` corresponds to `enumerated_slate_actions`.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward_factual: array-like, shape (n_rounds, len_list)
        When reward_structure='standard_additive', :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j \\neq k} W(a(k), a(j)))`.
        When reward_structure='cascade_additive', :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j < k} W(a(k), a(j)))`.
        Otherwise, :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j \\neq k} g^{-1}(f(x, a(j))) * W(k, j)`

    """
    is_additive = reward_structure in ["standard_additive", "cascade_additive"]
    is_cascade = reward_structure in ["cascade_additive", "cascade_decay"]

    n_rounds = context.shape[0]
    if is_enumerated:
        action = np.tile(action, n_rounds)

    action_2d = action.reshape((-1, len_list)).astype(int)
    n_enumerated_slate_actions = len(action) // n_rounds
    expected_reward = base_reward_function(
        context=context, action_context=action_context, random_state=random_state
    )

    if reward_type == "binary":
        expected_reward = logit(expected_reward)

    expected_reward_factual = np.zeros_like(action_2d, dtype="float16")
    for pos_ in np.arange(len_list):
        tmp_fixed_reward = expected_reward[
            np.arange(len(action_2d)) // n_enumerated_slate_actions,
            action_2d[:, pos_],
        ]

        if reward_structure == "independent":
            pass
        elif is_additive:
            for pos2_ in np.arange(len_list):
                if is_cascade:
                    if pos_ <= pos2_:
                        break
                elif pos_ == pos2_:
                    continue
                tmp_fixed_reward += action_interaction_weight_matrix[
                    action_2d[:, pos_], action_2d[:, pos2_]
                ]
        else:
            for pos2_ in np.arange(len_list):
                if is_cascade:
                    if pos_ <= pos2_:
                        break
                elif pos_ == pos2_:
                    continue
                expected_reward_ = expected_reward[
                    np.arange(len(action_2d)) // n_enumerated_slate_actions,
                    action_2d[:, pos2_],
                ]
                weight_ = action_interaction_weight_matrix[pos_, pos2_]
                tmp_fixed_reward += expected_reward_ * weight_
        expected_reward_factual[:, pos_] = tmp_fixed_reward

    if reward_type == "binary":
        expected_reward_factual = sigmoid(expected_reward_factual)
    else:
        expected_reward_factual = np.clip(expected_reward_factual, 0, None)

    return expected_reward_factual
