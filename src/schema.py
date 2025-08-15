from typing import Literal

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    random_state: int
    n_rounds: int
    n_unique_action: int
    len_list: int
    dim_context: int
    reward_type: Literal["binary"]
    reward_structure: Literal["standard", "cascade", "independent"]
    is_factorizable: bool
    eta: float
    lambda_: float
    interaction_function: Literal["additive", "decay"]
    decay_function: Literal["inverse"]
    n_random_state: int
    behavior_policy: Literal["linear"] | None = None
    click_model: Literal["cascade", "pbm"] | None = None
    behavior_policy_function: str | None = None

    @classmethod
    def load(cls, config_path: str):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)
