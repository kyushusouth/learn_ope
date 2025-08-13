from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

from dataset.synthetic import logistic_reward_function
from dataset.synthetic_slate import (
    SyntheticSlateBanditDataset,
    linear_behavior_policy_logit,
)
from ope.estimators_slate import (
    SelfNormalizedSlateIndependentIPS,
    SelfNormalizedSlateRewardInteractionIPS,
    SelfNormalizedSlateStandardIPS,
    SlateCascadeDoublyRobust,
    SlateIndependentIPS,
    SlateRewardInteractionIPS,
    SlateStandardIPS,
)
from ope.meta_slate import SlateOffPolicyEvaluation
from ope.regression_model_slate import SlateRegressionModel
from schema import Config

sns.set_style("whitegrid")


def main():
    cfg = Config.load(Path(__file__).parent.parent.joinpath("conf", "config.yaml"))
    result_dir = Path(__file__).parent.parent.joinpath("result")
    result_dir.mkdir(parents=True, exist_ok=True)

    ips = SlateStandardIPS(len_list=cfg.len_list, estimator_name="IPS")
    iips = SlateIndependentIPS(len_list=cfg.len_list, estimator_name="IIPS")
    rips = SlateRewardInteractionIPS(len_list=cfg.len_list, estimator_name="rips")
    sips = SelfNormalizedSlateStandardIPS(len_list=cfg.len_list, estimator_name="SIPS")
    siips = SelfNormalizedSlateIndependentIPS(
        len_list=cfg.len_list, estimator_name="SIIPS"
    )
    srips = SelfNormalizedSlateRewardInteractionIPS(
        len_list=cfg.len_list, estimator_name="SRIPS"
    )
    cascade_dr = SlateCascadeDoublyRobust(
        len_list=cfg.len_list,
        n_unique_action=cfg.n_unique_action,
        estimator_name="Cascade-DR",
    )

    regressor = SlateRegressionModel(
        DecisionTreeRegressor(max_depth=3, random_state=cfg.random_state),
        len_list=cfg.len_list,
        n_unique_action=cfg.n_unique_action,
        fitting_method="iw",
    )

    behavior_policy_function = None
    if cfg.behavior_policy_function == "linear":
        behavior_policy_function = linear_behavior_policy_logit
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=cfg.n_unique_action,
        len_list=cfg.len_list,
        dim_context=cfg.dim_context,
        reward_type=cfg.reward_type,
        reward_structure=cfg.reward_structure,
        click_model=cfg.click_model,
        behavior_policy_function=behavior_policy_function,
        base_reward_function=logistic_reward_function,
        random_state=cfg.random_state,
        is_factorizable=cfg.is_factorizable,
        eta=cfg.eta,
    )

    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=cfg.n_rounds, return_pscore_item_position=True
    )
    evaluation_policy = "similar" if cfg.lambda_ > 0 else "dissimilar"
    epsilon = 1 - abs(cfg.lambda_)
    if behavior_policy_function is None:
        behavior_policy_logit_ = np.ones((cfg.n_rounds, cfg.n_unique_action))
        evaluation_policy_logit_ = linear_behavior_policy_logit(
            context=bandit_feedback["context"],
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
    else:
        behavior_policy_logit_ = behavior_policy_function(
            context=bandit_feedback["context"],
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
        if evaluation_policy == "similar":
            evaluation_policy_logit_ = (
                1 - epsilon
            ) * behavior_policy_logit_ + epsilon * np.ones(behavior_policy_logit_.shape)
        else:
            evaluation_policy_logit_ = (
                1 - epsilon
            ) * -behavior_policy_logit_ + epsilon * np.ones(
                behavior_policy_logit_.shape
            )
    (
        bandit_feedback["evaluation_policy_pscore"],
        bandit_feedback["evaluation_policy_pscore_item_position"],
        bandit_feedback["evaluation_policy_pscore_cascade"],
    ) = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=bandit_feedback["action"],
        evaluation_policy_logit_=evaluation_policy_logit_,
        return_pscore_item_position=True,
    )
    bandit_feedback["ground_truth_policy_value"] = (
        dataset.calc_ground_truth_policy_value(
            evaluation_policy_logit_=evaluation_policy_logit_,
            context=bandit_feedback["context"],
        )
    )
    bandit_feedback["evaluation_policy_action_dist"] = (
        dataset.calc_evaluation_policy_action_dist(
            action=bandit_feedback["action"],
            evaluation_policy_logit_=evaluation_policy_logit_,
        )
    )

    ope = SlateOffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=[ips, iips, rips, sips, siips, srips, cascade_dr],
    )

    q_hat = regressor.fit_predict(
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        pscore_cascade=bandit_feedback["pscore_cascade"],
        evaluation_policy_pscore_cascade=bandit_feedback[
            "evaluation_policy_pscore_cascade"
        ],
        evaluation_policy_action_dist=bandit_feedback["evaluation_policy_action_dist"],
    )

    se_dict_ = ope.evaluate_performance_of_estimators(
        ground_truth_policy_value=bandit_feedback["ground_truth_policy_value"],
        evaluation_policy_pscore=bandit_feedback["evaluation_policy_pscore"],
        evaluation_policy_pscore_item_position=bandit_feedback[
            "evaluation_policy_pscore_item_position"
        ],
        evaluation_policy_pscore_cascade=bandit_feedback[
            "evaluation_policy_pscore_cascade"
        ],
        evaluation_policy_action_dist=bandit_feedback["evaluation_policy_action_dist"],
        q_hat=q_hat,
        metric="se",
    )
    relative_ee_dict_ = ope.evaluate_performance_of_estimators(
        ground_truth_policy_value=bandit_feedback["ground_truth_policy_value"],
        evaluation_policy_pscore=bandit_feedback["evaluation_policy_pscore"],
        evaluation_policy_pscore_item_position=bandit_feedback[
            "evaluation_policy_pscore_item_position"
        ],
        evaluation_policy_pscore_cascade=bandit_feedback[
            "evaluation_policy_pscore_cascade"
        ],
        evaluation_policy_action_dist=bandit_feedback["evaluation_policy_action_dist"],
        q_hat=q_hat,
        metric="relative-ee",
    )

    se_df = pd.DataFrame(data={"estimator": se_dict_.keys(), "se": se_dict_.values()})
    relative_ee_df = pd.DataFrame(
        data={
            "estimator": relative_ee_dict_.keys(),
            "relative_ee": relative_ee_dict_.values(),
        }
    )
    metrics_df = se_df.merge(relative_ee_df, how="left", on="estimator")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    sns.barplot(metrics_df, x="estimator", y="se", ax=ax[0])
    sns.barplot(metrics_df, x="estimator", y="relative_ee", ax=ax[1])
    fig.tight_layout()
    fig.savefig(result_dir.joinpath("metrics.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
