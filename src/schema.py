from pathlib import Path

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    seed: int

    @classmethod
    def load(cls, config_path: str):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)
