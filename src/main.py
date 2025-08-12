from itertools import permutations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset.synthetic import logistic_reward_function
from dataset.synthtic_slate import (
    SyntheticSlateBanditDataset,
    linear_behavior_policy_logit,
)
from schema import Config


def main():
    cfg = Config.load(Path(__file__).parent.parent.joinpath("conf", "config.yaml"))

    rand_gen = np.random.default_rng(cfg.seed)

    n_rounds = 10000
    dim_context = 8
    n_unique_action = 10
    dim_action_context = 8
    tau = 1
    context = rand_gen.normal(0, 1, (n_rounds, dim_context))
    action_context = rand_gen.normal(0, 1, (n_unique_action, dim_action_context))
    len_list = 10

    dataset = SyntheticSlateBanditDataset(
        n_unique_action=10,
        dim_context=5,
        len_list=3,
        base_reward_function=logistic_reward_function,
        behavior_policy_function=linear_behavior_policy_logit,
        reward_type="binary",
        reward_structure="cascade_additive",
        click_model="cascade",
        random_state=12345,
        is_factorizable=False,
        eta=1.0,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=True, clip_logit_value=None
    )


if __name__ == "__main__":
    main()
