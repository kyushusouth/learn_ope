from pathlib import Path

import numpy as np

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

    behavior_policy_function = linear_behavior_policy_logit(
        context=context,
        action_context=action_context,
        random_state=cfg.seed,
        tau=tau,
    )

    rewards = logistic_reward_function(context=context, action_context=action_context)


if __name__ == "__main__":
    main()
