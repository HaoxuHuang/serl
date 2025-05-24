import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class BasketballEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    # TO DO
