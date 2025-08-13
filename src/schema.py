import yaml
from pydantic import BaseModel


class Config(BaseModel):
    random_state: int
    n_rounds: int
    n_unique_action: int
    len_list: int
    dim_context: int
    behavior_policy_function: str
    reward_type: str
    reward_structure: str
    click_model: str
    is_factorizable: bool
    eta: float
    lambda_: float

    @classmethod
    def load(cls, config_path: str):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)
