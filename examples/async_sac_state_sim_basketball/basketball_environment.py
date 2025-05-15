import gym
from gym import spaces
import mujoco
import numpy as np


class FrankaBasketEnv(gym.Env):
    def __init__(self, xml_path="franka_basket.xml"):
        super().__init__()
        # 加载模型与数据
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 观测空间：7 关节位姿＋速度＋球的位置速度
        obs_dim = 7 * 2 + 6
        self.observation_space = spaces.Box(-np.inf,
                                            np.inf, shape=(obs_dim,), dtype=np.float32)
        # 动作空间：7 维扭矩
        self.action_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)

        # 篮筐中心，用于奖励计算
        self.hoop_center = np.array([1.2, 0.0, 1.0])

    def reset(self):
        # 恢复到初始状态
        mujoco.mj_resetData(self.model, self.data)
        # 随机放球
        self.data.xpos[self.model.body('ball').id] = [
            0.7, np.random.uniform(-0.1, 0.1), 1.2]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        # 将动作映射到真实扭矩
        low, high = np.array(self.model.actuator_ctrlrange[:, 0]), np.array(
            self.model.actuator_ctrlrange[:, 1])
        torque = low + (action + 1) / 2 * (high - low)
        self.data.ctrl[:] = torque

        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        self.step_count += 1
        done = self.step_count >= 200 or self._is_scored()
        return obs, reward, done, {}

    def _get_obs(self):
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()
        ball_pos = self.data.xpos[self.model.body('ball').id].copy()
        ball_vel = self.data.xvel[self.model.body('ball').id].copy()
        return np.concatenate([qpos, qvel, ball_pos, ball_vel])

    def _compute_reward(self, obs):
        pos = obs[-6:-3]
        dist = np.linalg.norm(pos - self.hoop_center)
        return 100.0 if self._is_scored() else -dist

    def _is_scored(self):
        pos = self.data.xpos[self.model.body('ball').id]
        z, xy = pos[2], pos[:2]
        return (0.95 < z < 1.05) and (np.linalg.norm(xy - self.hoop_center[:2]) < 0.45)

    def render(self, mode="human"):
        # 只要在外部用 viewer.launch，就会自动渲染
        pass
