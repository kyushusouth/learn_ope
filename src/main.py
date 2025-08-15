from itertools import product
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


def experiment(cfg: Config) -> dict[str, float]:
    evaluation_policy = "similar" if cfg.lambda_ > 0 else "dissimilar"
    epsilon = 1 - abs(cfg.lambda_)

    ips = SlateStandardIPS(len_list=cfg.len_list, estimator_name="IPS")
    iips = SlateIndependentIPS(len_list=cfg.len_list, estimator_name="IIPS")
    rips = SlateRewardInteractionIPS(len_list=cfg.len_list, estimator_name="RIPS")
    sips = SelfNormalizedSlateStandardIPS(len_list=cfg.len_list, estimator_name="SIPS")
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
        decay_function=cfg.decay_function,
        eta=cfg.eta,
    )

    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=cfg.n_rounds, return_pscore_item_position=True
    )
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
        ope_estimators=[ips, iips, rips, sips, cascade_dr],
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
    return se_dict_


def main():
    cfg = Config.load(Path(__file__).parent.parent.joinpath("conf", "config.yaml"))
    result_dir = Path(__file__).parent.parent.joinpath("result")
    result_dir.mkdir(parents=True, exist_ok=True)

    random_state_ptn = range(cfg.n_random_state)
    reward_structure_ptn = ["standard_additive", "cascade_additive", "independent"]

    def experiment_n_rounds():
        n_rounds_ptn = [250, 500, 1000, 2000, 4000]
        metrics_dfs = []
        for random_state, reward_structure, n_rounds in product(
            random_state_ptn, reward_structure_ptn, n_rounds_ptn
        ):
            print(f"{random_state=}, {reward_structure=}, {n_rounds=}")
            cfg.random_state = random_state
            cfg.reward_structure = reward_structure
            cfg.n_rounds = n_rounds

            se_dict_ = experiment(cfg)
            metrics_df = pd.DataFrame(
                data={"estimator": se_dict_.keys(), "se": se_dict_.values()}
            )
            metrics_df["random_state"] = cfg.random_state
            metrics_df["reward_structure"] = cfg.reward_structure
            metrics_df["n_rounds"] = cfg.n_rounds
            metrics_dfs.append(metrics_df)

        metrics_df = pd.concat(metrics_dfs, axis=0)
        metrics_df = (
            metrics_df.groupby(["estimator", "reward_structure", "n_rounds"])
            .agg(mse=("se", "mean"))
            .reset_index()
        )

        for reward_structure in reward_structure_ptn:
            plt.figure(figsize=(12, 8))
            sns.lineplot(
                metrics_df.query("reward_structure == @reward_structure"),
                x="n_rounds",
                y="mse",
                hue="estimator",
            )
            plt.tight_layout()
            plt.savefig(result_dir.joinpath(f"n_rounds_{reward_structure}.png"))
            plt.close()

    def experiment_len_list():
        len_list_ptn = [3, 4, 5]
        metrics_dfs = []
        for random_state, reward_structure, len_list in product(
            random_state_ptn, reward_structure_ptn, len_list_ptn
        ):
            print(f"{random_state=}, {reward_structure=}, {len_list=}")
            cfg.random_state = random_state
            cfg.reward_structure = reward_structure
            cfg.len_list = len_list

            se_dict_ = experiment(cfg)
            metrics_df = pd.DataFrame(
                data={"estimator": se_dict_.keys(), "se": se_dict_.values()}
            )
            metrics_df["random_state"] = cfg.random_state
            metrics_df["reward_structure"] = cfg.reward_structure
            metrics_df["len_list"] = cfg.len_list
            metrics_dfs.append(metrics_df)

        metrics_df = pd.concat(metrics_dfs, axis=0)
        metrics_df = (
            metrics_df.groupby(["estimator", "reward_structure", "len_list"])
            .agg(mse=("se", "mean"))
            .reset_index()
        )

        for reward_structure in reward_structure_ptn:
            plt.figure(figsize=(12, 8))
            sns.lineplot(
                metrics_df.query("reward_structure == @reward_structure"),
                x="len_list",
                y="mse",
                hue="estimator",
            )
            plt.tight_layout()
            plt.savefig(result_dir.joinpath(f"len_list_{reward_structure}.png"))
            plt.close()

    def experiment_lambda_():
        lambda__ptn = [-0.8, -0.4, 0, 0.4, 0.8]
        metrics_dfs = []
        for random_state, reward_structure, lambda_ in product(
            random_state_ptn, reward_structure_ptn, lambda__ptn
        ):
            print(f"{random_state=}, {reward_structure=}, {lambda_=}")
            cfg.random_state = random_state
            cfg.reward_structure = reward_structure
            cfg.lambda_ = lambda_

            se_dict_ = experiment(cfg)
            metrics_df = pd.DataFrame(
                data={"estimator": se_dict_.keys(), "se": se_dict_.values()}
            )
            metrics_df["random_state"] = cfg.random_state
            metrics_df["reward_structure"] = cfg.reward_structure
            metrics_df["lambda_"] = cfg.lambda_
            metrics_dfs.append(metrics_df)

        metrics_df = pd.concat(metrics_dfs, axis=0)
        metrics_df = (
            metrics_df.groupby(["estimator", "reward_structure", "lambda_"])
            .agg(mse=("se", "mean"))
            .reset_index()
        )

        for reward_structure in reward_structure_ptn:
            plt.figure(figsize=(12, 8))
            sns.lineplot(
                metrics_df.query("reward_structure == @reward_structure"),
                x="lambda_",
                y="mse",
                hue="estimator",
            )
            plt.tight_layout()
            plt.savefig(result_dir.joinpath(f"lambda_{reward_structure}.png"))
            plt.close()

    experiment_n_rounds()
    experiment_len_list()
    experiment_lambda_()


if __name__ == "__main__":
    main()
