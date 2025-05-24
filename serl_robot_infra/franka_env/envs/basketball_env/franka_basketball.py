import numpy as np
import gym
import time
import requests
import copy

from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.basketball_env.config import BasketballEnvConfig


class FrankaBasketball(FrankaEnv):
    def compute_reward(self, obs):
        """
        Reward from camera image.
        Alignment camera frequency 30Hz with policy frequency 50Hz.
        Individual process for camera. Communicate with shared memory or network or whatever.
        """
        pass

    def go_to_rest(self, joint_reset=False):
        # stop
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.5)

        # reset
        super().go_to_rest(joint_reset)

    def init_cameras(self):
        """
        USB Camera can be accessed through v4l device.
        """
        pass

    def close_cameras(self):
        pass
