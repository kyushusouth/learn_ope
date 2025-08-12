from pathlib import Path

import numpy as np

from dataset.synthetic import logistic_reward_function
from dataset.synthtic_slate import (
    SyntheticSlateBanditDataset,
    linear_behavior_policy_logit,
)
from ope.estimators_slate import (
    SlateIndependentIPS,
    SlateRewardInteractionIPS,
    SlateStandardIPS,
)
from ope.meta_slate import SlateOffPolicyEvaluation
from schema import Config


def main():
    cfg = Config.load(Path(__file__).parent.parent.joinpath("conf", "config.yaml"))

    n_rounds = 10000
    n_unique_action = 10
    len_list = 3
    dim_contect = 2
    random_state = 42

    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_contect,
        reward_type="binary",
        reward_structure="cascade_additive",
        click_model="cascade",
        behavior_policy_function=linear_behavior_policy_logit,
        base_reward_function=logistic_reward_function,
        random_state=random_state,
        is_factorizable=False,
        eta=1.0,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=True
    )
    policy_value = dataset.calc_on_policy_policy_value(
        reward=bandit_feedback["reward"],
        slate_id=bandit_feedback["slate_id"],
    )

    random_policy_logit_ = np.zeros((n_rounds, n_unique_action))
    base_expected_reward = dataset.base_reward_function(
        context=bandit_feedback["context"],
        action_context=dataset.action_context,
        random_state=random_state,
    )
    optimal_policy_logit_ = base_expected_reward * 3

    random_policy_pscores = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=bandit_feedback["action"],
        evaluation_policy_logit_=random_policy_logit_,
    )
    optimal_policy_pscores = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=bandit_feedback["action"],
        evaluation_policy_logit_=optimal_policy_logit_,
    )

    ope = SlateOffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=[
            SlateStandardIPS(len_list=len_list),
            SlateIndependentIPS(len_list=len_list),
            SlateRewardInteractionIPS(len_list=len_list),
        ],
    )

    _, estimated_interval_random = ope.summarize_off_policy_estimates(
        evaluation_policy_pscore=random_policy_pscores[0],
        evaluation_policy_pscore_item_position=random_policy_pscores[1],
        evaluation_policy_pscore_cascade=random_policy_pscores[2],
        alpha=0.05,
        n_bootstrap_samples=1000,
        random_state=random_state,
    )
    estimated_interval_random["policy_name"] = "random"

    _, estimated_interval_optimal = ope.summarize_off_policy_estimates(
        evaluation_policy_pscore=optimal_policy_pscores[0],
        evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
        evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
        alpha=0.05,
        n_bootstrap_samples=1000,
        random_state=dataset.random_state,
    )
    estimated_interval_optimal["policy_name"] = "optimal"

    ground_truth_policy_value_random = dataset.calc_ground_truth_policy_value(
        context=bandit_feedback["context"],
        evaluation_policy_logit_=random_policy_logit_,
    )
    ground_truth_policy_value_optimal = dataset.calc_ground_truth_policy_value(
        context=bandit_feedback["context"],
        evaluation_policy_logit_=optimal_policy_logit_,
    )
    estimated_interval_random["ground_truth"] = ground_truth_policy_value_random
    estimated_interval_optimal["ground_truth"] = ground_truth_policy_value_optimal

    relative_ee_for_random_evaluation_policy = ope.summarize_estimators_comparison(
        ground_truth_policy_value=ground_truth_policy_value_random,
        evaluation_policy_pscore=random_policy_pscores[0],
        evaluation_policy_pscore_item_position=random_policy_pscores[1],
        evaluation_policy_pscore_cascade=random_policy_pscores[2],
    )
    relative_ee_for_optimal_evaluation_policy = ope.summarize_estimators_comparison(
        ground_truth_policy_value=ground_truth_policy_value_optimal,
        evaluation_policy_pscore=optimal_policy_pscores[0],
        evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
        evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
    )

    print(relative_ee_for_random_evaluation_policy)
    print(relative_ee_for_optimal_evaluation_policy)

    # need to evaluate variance


if __name__ == "__main__":
    main()
