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
        self.episode_counter = 0

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """重置环境并清空历史记录"""
        obs, info = super().reset(**kwargs)
        self.reward_history = defaultdict(list)
        self.observation_history = [obs]  # 初始观测
        self.episode_counter += 1
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行动作，记录reward和观测"""
        obs, rew, terminated, truncated, info = super().step(action)
        for k, v in info['reward'].items():
            self.reward_history[k].append(v)
        self.observation_history.append(obs)
        return obs, rew, terminated, truncated, info

    def plot_rewards(self) -> None:
        """绘制当前episode的reward曲线"""
        plt.figure()
        for reward_key, reward_value in self.reward_history:
            plt.plot(reward_value, label=reward_key)
        plt.title(f"Reward History (Episode {self.episode_counter})")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()

    def plot_observation_component(self, key_path: List[str]) -> None:
        """
        绘制观测中指定字段的变化
        :param key_path: 观测字段路径，例如 ['state', 'panda/tcp_pos']
        """
        values = []
        for obs in self.observation_history:
            current = obs
            for key in key_path:
                current = current[key]
            values.append(current)
        values = np.array(values)

        plt.figure()
        if len(values.shape) == 1:
            plt.plot(values, label=key_path[-1])
        else:
            for dim in range(values.shape[1]):
                plt.plot(values[:, dim], label=f"{key_path[-1]} (dim {dim})")
        plt.title("Observation: " + " -> ".join(key_path))
        plt.xlabel("Step")
        plt.legend()
        plt.show()