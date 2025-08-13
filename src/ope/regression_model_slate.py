from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone


@dataclass
class SlateRegressionModel(BaseEstimator):
    base_model: BaseEstimator
    len_list: int
    n_unique_action: int
    fitting_method: str = "normal"

    def __post_init__(self):
        self.base_model_list = [clone(self.base_model) for _ in range(self.len_list)]
        self.action_context = np.eye(self.n_unique_action)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
    ):
        """Fit the regression model on given logged bandit data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, (n_rounds * len_list,)
            Actions observed at each slot in a ranking/slate in logged bandit data, i.e., :math:`a_{i}(l)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds * len_list,)
            Slot-level rewards observed for each data in logged bandit data, i.e., :math:`r_{i}(l)`.

        pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`).

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_action_dist: array-like (n_rounds * len_list * n_unique_actions, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e({a'}_t(k) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        """
        # (n_rounds_ * len_list, ) -> (n_rounds_, len_list)
        action = action.reshape((-1, self.len_list))
        reward = reward.reshape((-1, self.len_list))
        iw = (evaluation_policy_pscore_cascade / pscore_cascade).reshape(
            (-1, self.len_list)
        )

        # (n_rounds_, )
        n_rounds_ = len(action)
        sample_weight = np.ones(n_rounds_)

        for pos_ in range(self.len_list)[::-1]:
            X, y = self._preprocess_for_reg_model(
                context=context,
                action=action,
                reward=reward,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
                position_=pos_,
            )

            if self.fitting_method == "iw":
                sample_weight = iw[:, pos_]

            self.base_model_list[pos_].fit(X, y, sample_weight=sample_weight)

    def predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
    ):
        """Predict the Q function values.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        action: array-like, shape (n_rounds_of_new_data * len_list, )
            Action vectors for new data.

        Returns
        -----------
        q_hat: array-like, shape (n_rounds_of_new_data * len_list * n_unique_action, )
            Estimated Q function values of new data.
            :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.

        """
        n_rounds_of_new_data = len(context)
        # (n_rounds_of_new_data * len_list, ) -> (n_rounds_of_new_data, len_list)
        action = action.reshape((-1, self.len_list))
        # (n_rounds_, len_list, n_unique_action, )
        q_hat = np.zeros((n_rounds_of_new_data, self.len_list, self.n_unique_action))
        for pos_ in range(self.len_list)[::-1]:
            # the action vector shrinks every time as the position_ decreases
            # (n_rounds_of_new_data, position_ - 1)
            action = action[:, :pos_]
            # (n_rounds_of_new_data, dim_context) -> (n_rounds_of_new_data * n_unique_action, dim_context)
            context_ = []
            # (n_rounds_of_new_data, position_) -> (n_rounds_of_new_data * n_unique_action, position_)
            action_ = []
            for i in range(n_rounds_of_new_data):
                for a_ in range(self.n_unique_action):
                    context_.append(context[i])
                    action_.append(np.append(action[i], a_))
            # (n_rounds_of_new_data * n_unique_action, dim_context + position_)
            X = np.concatenate([context_, action_], axis=1)
            # (n_rounds_of_new_data * n_unique_action, ) -> (n_rounds_of_new_data, n_unique_action)
            q_hat[:, pos_, :] = (
                self.base_model_list[pos_]
                .predict(X)
                .reshape((-1, self.n_unique_action))
            )
        # (n_rounds_of_new_data * len_list * n_unique_action, )
        return q_hat.flatten()

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
    ):
        """Fit the regression model on given logged bandit data and predict the Q function values on the same data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, (n_rounds * len_list,)
            Actions observed at each slot in a ranking/slate in logged bandit data, i.e., :math:`a_{i}(l)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds * len_list,)
            Slot-level rewards observed for each data in logged bandit data, i.e., :math:`r_{i}(l)`.

        pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`).

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_action_dist: array-like (n_rounds * len_list * n_unique_actions, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        Returns
        -----------
        q_hat: array-like, shape (n_rounds_of_new_data * len_list * n_unique_action, )
            Estimated Q functions for new data by the regression model.

        """
        self.fit(
            context=context,
            action=action,
            reward=reward,
            pscore_cascade=pscore_cascade,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
        # (n_rounds_test, len_list, n_unique_action, )
        return self.predict(context=context, action=action)

    def _preprocess_for_reg_model(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        position_: int,
    ):
        """Preprocess feature vectors and target variables for training a regression model.

        Note
        -----
        Please override this method if you want to use another feature enginnering
        for training the regression model.

        Parameters
        -----------
        context: array-like, shape (n_rounds_, dim_context)
            Context vectors in the training set of logged bandit data.

        action: array-like, (n_rounds_ * len_list, )
            Actions observed at each slot in a ranking/slate in logged bandit data, i.e., :math:`a_{i}(l)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds_ * len_list, )
            Slot-level rewards observed for each data in logged bandit data, i.e., :math:`r_{i}(l)`.

        evaluation_policy_action_dist: array-like (n_rounds_ * len_list * n_unique_actions, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        position_: int
            Position id (slot) in a slate.

        Returns
        -----------
        X, y: array-like, shape(n_rounds, )
            Input and target vectors in prediction.

        """
        n_rounds_ = len(context)
        # (n_rounds_, len_list) -> (n_rounds_, position_)
        action = action[:, : position_ + 1]
        # (n_rounds_, len_list) -> (n_rounds_, )
        reward = reward[:, position_]
        # estimator input
        X = np.concatenate([context, action], axis=1)
        # estimate the Q function at the next position
        # (n_rounds_, )
        if position_ + 1 == self.len_list:
            q_hat_at_next_position = np.zeros(n_rounds_)
        else:
            # (n_rounds_ * len_list * n_unique_action, ) -> (n_rounds_, len_list, n_unique_action) -> (n_rounds_, len_list) -> (n_rounds_ * n_unique_action, )
            evaluation_policy_action_dist_at_next_position = (
                evaluation_policy_action_dist.reshape(
                    (-1, self.len_list, self.n_unique_action)
                )[:, position_ + 1, :]
            ).flatten()
            # (n_rounds_, dim_context) -> (n_rounds_ * n_unique_action, dim_context)
            context_ = []
            # (n_rounds_, position_ + 1) -> (n_rounds_ * n_unique_action, position_ + 1)
            action_ = []
            for i in range(n_rounds_):
                for a_ in range(self.n_unique_action):
                    context_.append(context[i])
                    action_.append(np.append(action[i], a_))
            X_ = np.concatenate([context_, action_], axis=1)
            # (n_rounds_ * n_unique_action, ) -> (n_rounds_, )
            q_hat_at_next_position = self.base_model_list[position_ + 1].predict(X_)
            # the expected Q function under the evaluation policy
            # (n_rounds_ * n_unique_action, ) -> (n_rounds_, n_unique_action) -> (n_rounds_, )
            q_hat_at_next_position = (
                (
                    evaluation_policy_action_dist_at_next_position
                    * q_hat_at_next_position
                )
                .reshape((-1, self.n_unique_action))
                .sum(axis=1)
            )
        # (n_rounds_, )
        y = reward + q_hat_at_next_position
        # (n_rounds_, dim_context + position_), (n_rounds_, )
        return X, y
