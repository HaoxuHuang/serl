import numpy as np
from typing import Dict


class BasketballEnvConfig:
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://172.16.0.3:5000/"
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = 0.2  # choose a small value for safety
    RESET_POSE = np.array((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}