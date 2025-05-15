from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gym
import mujoco
import numpy as np
from gym import spaces

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "franka_basket.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])


class FrankaBasketEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: float = 0.1,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        target_pos: np.ndarray = None,
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        # default target at geometric center of workspace
        self._target_pos = (
            np.mean(_CARTESIAN_BOUNDS, axis=0)
            if target_pos is None
            else target_pos
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs

        # Caching
        self._panda_dof_ids = np.asarray([
            self._model.joint(f"joint{i}").id for i in range(1, 8)
        ])
        self._panda_ctrl_ids = np.asarray([
            self._model.actuator(f"actuator{i}").id for i in range(1, 8)
        ])
        self._block_geom_id = self._model.geom("block").id
        self._block_z = self._model.geom("block").size[2]

        # Observation space
        state_spaces = {
            "panda/tcp_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            # "panda/tcp_vel": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        }
        if not self.image_obs:
            state_spaces["block_pos"] = spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            )

        obs_dict = {"state": spaces.Dict(state_spaces)}
        if self.image_obs:
            obs_dict["images"] = spaces.Dict({
                "front": spaces.Box(
                    low=0, high=255,
                    shape=(render_spec.height, render_spec.width, 3),
                    dtype=np.uint8,
                ),
                "wrist": spaces.Box(
                    low=0, high=255,
                    shape=(render_spec.height, render_spec.width, 3),
                    dtype=np.uint8,
                ),
            })

        self.observation_space = spaces.Dict(obs_dict)

        # Action space: x, y, z increments
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        self._viewer = MujocoRenderer(self.model, self.data)
        self._viewer.render(self.render_mode)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Initialize target
        self._current_target = self._target_pos.copy()

        # Sample new block position
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache block heights from joint
        block_qpos = self._data.jnt("block").qpos
        self._z_init = block_qpos[2]
        self._z_success = self._z_init + 0.2

        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        dx, dy, dz = action

        # Update target
        self._current_target = np.clip(
            self._current_target + np.asarray([dx, dy, dz]) * self._action_scale,
            *_CARTESIAN_BOUNDS,
        )

        # IK control
        for _ in range(self._n_substeps):
            # print("xxxx", self._data.mocap_quat)
            tau = opspace(
                model=self._model,
                data=self._data,
                pos=self._current_target,
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        done = self.time_limit_exceeded()
        return obs, rew, done, False, {}

    def render(self):
        return [
            self._viewer.render(render_mode="rgb_array", camera_id=i)
            for i in self.camera_id
        ]

    def _compute_observation(self) -> dict:
        obs = {"state": {}}
        obs_state = obs["state"]
        obs_state["panda/tcp_pos"] = self._current_target.astype(np.float32)
        # obs_state["panda/tcp_vel"] = self._data.sensor(
        #     "2f85/pinch_vel").data.astype(np.float32)
        if not self.image_obs:
            # block position from joint qpos
            obs_state["block_pos"] = self._data.jnt("block").qpos[:3].astype(np.float32)
        else:
            imgs = self.render()
            obs["images"] = {"front": imgs[0], "wrist": imgs[1]}
        if self.render_mode == "human":
            self._viewer.render(self.render_mode)
        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.jnt("block").qpos[:3]
        tcp_pos = self._current_target
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        r_lift = np.clip(r_lift, 0.0, 1.0)
        return 0.3 * r_close + 0.7 * r_lift


if __name__ == "__main__":
    env = FrankaBasketEnv(render_mode="human")
    env.reset()
    for _ in range(100):
        env.step(np.random.uniform(-1, 1, 3))
        env.render()
    env.close()
