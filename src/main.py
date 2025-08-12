from pathlib import Path

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

    dataset = SyntheticSlateBanditDataset(
        n_unique_action=10,
        len_list=3,
        dim_context=2,
        reward_type="binary",
        reward_structure="cascade_additive",
        click_model="cascade",
        behavior_policy_function=linear_behavior_policy_logit,
        base_reward_function=logistic_reward_function,
        random_state=12345,
        is_factorizable=False,
        eta=1.0,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=True
    )

    random_dataset = SyntheticSlateBanditDataset(
        n_unique_action=10,
        len_list=3,
        dim_context=2,
        reward_type="binary",
        reward_structure="cascade_additive",
        click_model="cascade",
        behavior_policy_function=None,  # set to uniform random
        base_reward_function=logistic_reward_function,
        random_state=12345,
        is_factorizable=False,
        eta=1.0,
    )
    random_feedback = random_dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=True
    )

    ope = SlateOffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=[
            SlateStandardIPS(len_list=3),
            SlateIndependentIPS(len_list=3),
            SlateRewardInteractionIPS(len_list=3),
        ],
    )
    estimated_policy_value = ope.estimate_policy_values(
        evaluation_policy_pscore=random_feedback["pscore"],
        evaluation_policy_pscore_item_position=random_feedback["pscore_item_position"],
        evaluation_policy_pscore_cascade=random_feedback["pscore_cascade"],
    )
    print(estimated_policy_value)


if __name__ == "__main__":
    main()
