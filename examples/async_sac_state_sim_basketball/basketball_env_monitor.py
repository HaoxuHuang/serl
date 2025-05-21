import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

class BasketMonitorWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reward_history = defaultdict(list)
        self.observation_history = []
        self.action_history = []
        self.episode_counter = 0

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """重置环境并清空历史记录"""
        obs, info = super().reset(**kwargs)
        self.reward_history = defaultdict(list)
        self.observation_history = [obs]  # 初始观测
        self.action_history = []
        self.episode_counter += 1
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行动作，记录reward和观测"""
        obs, rew, terminated, truncated, info = super().step(action)
        self.action_history.append(action)
        for k, v in info['reward'].items():
            self.reward_history[k].append(v)
        self.observation_history.append(obs)
        return obs, rew, terminated, truncated, info

    def plot_rewards(self) -> None:
        """绘制当前episode的reward曲线"""
        plt.figure()
        for reward_key, reward_value in self.reward_history.items():
            plt.plot(reward_value, label=reward_key)
        plt.title(f"Reward History (Episode {self.episode_counter})")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()

    def plot_observation_component(self) -> None:
        """
        绘制观测的变化
        """
        obs = np.stack(self.observation_history)
        joint_pos = obs[:,:7]
        joint_vel = obs[:,7:14]
        tcp_pos = obs[:,14:17]
        tcp_vel = obs[:, 17:]

        plt.figure()
        # plot joint position
        plt.subplot(221)
        self.plot_observation_array(joint_pos, 'joint position')
        plt.subplot(222)
        self.plot_observation_array(joint_vel, 'joint velocity')
        plt.subplot(223)
        self.plot_observation_array(tcp_pos, 'tcp position')
        plt.subplot(224)
        self.plot_observation_array(tcp_vel, 'tcp velocity')
        plt.show()

    def plot_actions(self) -> None:
        plt.figure()
        actions = np.stack(self.action_history)
        for dim in range(actions.shape[1]):
            plt.plot(actions[:, dim], label=f"joint {dim}")
        plt.title("Action")
        plt.xlabel("Step")
        plt.legend()
        plt.show()


    def plot_observation_array(self, array, key: str):
        for dim in range(array.shape[1]):
            plt.plot(array[:, dim], label=f"{key} {dim}")
        plt.title("Observation: " + key)
        plt.xlabel("Step")
        plt.legend()