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

from franka_sim.controllers import jointspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])


class BasketEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        # action_scale: np.ndarray = np.asarray([0.1, 1]),
        action_scale: float = 0.05,
        angle_penalty: float = 0.00003,
        energy_penalty: float = 0.0001,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        self._action_scale = action_scale
        self._angle_penalty = angle_penalty
        self._energy_penalty = energy_penalty

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        # self._ctrl_id = self._model.actuator("actuator7").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        self.block_id = self._model.geom("block").id
        self.floor_id = self._model.geom("floor").id
        self.circle_o = self._model.site("visual_circle").pos[:2]
        self.circle_r = self._model.site("visual_circle").size[0]

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "panda/tcp_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "panda/tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        # "panda/gripper_pos": spaces.Box(
                        #     -np.inf, np.inf, shape=(1,), dtype=np.float32
                        # ),
                        "panda/joint_pos": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        "panda/joint_vel": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        # "panda/joint_torque": specs.Array(shape=(21,), dtype=np.float32),
                        # "panda/wrist_force": specs.Array(shape=(3,), dtype=np.float32),
                        # "block_pos": spaces.Box(
                        #     -np.inf, np.inf, shape=(3,), dtype=np.float32
                        # ),
                    }
                ),
            }
        )

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "panda/tcp_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/tcp_vel": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/gripper_pos": spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            "front": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0] * 7),
            high=np.asarray([1.0] * 7),
            dtype=np.float32,
        )

        # NOTE: gymnasium is used here since MujocoRenderer is not available in gym. It
        # is possible to add a similar viewer feature with gym, but that can be a future TODO
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._viewer = MujocoRenderer(
            self.model,
            self.data,
        )
        self._viewer.render(self.render_mode)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("panda/hand_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        # block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (0.48670042, 0.00820504, 0.610814)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        # self._z_init = self._data.sensor("block_pos").data[2]
        # self._z_success = self._z_init + 0.2

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray delta joint position

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        # x, y, z, grasp = action

        # # Set the mocap position.
        # pos = self._data.mocap_pos[0].copy()
        # dpos = np.asarray([x, y, z]) * self._action_scale[0]
        # npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        # self._data.mocap_pos[0] = npos


        # Set gripper grasp.
        # g = self._data.ctrl[self._ctrl_id] / 255
        # dg = grasp * self._action_scale[1]
        # ng = np.clip(g + dg, 0.0, 1.0)
        # self._data.ctrl[self._ctrl_id] = ng * 255
        pos = np.stack(
            [self._data.sensor(f"panda/joint{i}_pos").data for i in range(1, 8)],
            ).ravel()
        for _ in range(self._n_substeps):
            tau = jointspace(
                model=self._model,
                data=self._data,
                dof_ids=self._panda_dof_ids,
                joint=pos + action * self._action_scale,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        # for _ in range(self._n_substeps):
        # # for _ in range(1):
        #     pos = np.stack(
        #         [self._data.sensor(f"panda/joint{i}_pos").data for i in range(1, 8)],
        #     ).ravel()
        #     npos = pos + action * self._action_scale / self._n_substeps
        #     self._data.ctrl[self._panda_ctrl_ids] = npos
        #     mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward(action)
        terminated = self._compute_terminated()
        return obs, rew, terminated, False, {}

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("panda/hand_pos").data
        obs["state"]["panda/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self._data.sensor("panda/hand_vel").data
        obs["state"]["panda/tcp_vel"] = tcp_vel.astype(np.float32)

        # gripper_pos = np.array(
        #     self._data.ctrl[self._ctrl_id] / 255, dtype=np.float32
        # )
        # obs["state"]["panda/gripper_pos"] = gripper_pos

        joint_pos = np.stack(
            [self._data.sensor(f"panda/joint{i}_pos").data for i in range(1, 8)],
        ).ravel()
        obs["state"]["panda/joint_pos"] = joint_pos.astype(np.float32)

        joint_vel = np.stack(
            [self._data.sensor(f"panda/joint{i}_vel").data for i in range(1, 8)],
        ).ravel()
        obs["state"]["panda/joint_vel"] = joint_vel.astype(np.float32)

        # joint_torque = np.stack(
        # [self._data.sensor(f"panda/joint{i}_torque").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_torque"] = symlog(joint_torque.astype(np.float32))

        # wrist_force = self._data.sensor("panda/wrist_force").data.astype(np.float32)
        # obs["panda/wrist_force"] = symlog(wrist_force.astype(np.float32))

        # if self.image_obs:
        #     obs["images"] = {}
        #     obs["images"]["front"], obs["images"]["wrist"] = self.render()
        # else:
        #     block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        #     obs["state"]["block_pos"] = block_pos

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    def _compute_reward(self, action) -> float:
        # block_pos = self._data.sensor("block_pos").data
        # tcp_pos = self._data.sensor("panda/hand_pos").data
        # dist = np.linalg.norm(block_pos - tcp_pos)
        # r_close = np.exp(-20 * dist)
        # r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        # r_lift = np.clip(r_lift, 0.0, 1.0)
        # rew = 0.3 * r_close + 0.7 * r_lift
        # return rew
        
        pos = np.stack(
            [self._data.sensor(f"panda/joint{i}_pos").data for i in range(1, 8)],
        ).ravel()
        rew = -self._angle_penalty * (np.abs(pos - _PANDA_HOME)).sum()
        rew += -self._energy_penalty * np.linalg.norm(action)
        contact = self._data.contact
        for i in range(len(contact.geom1)):
            if (contact.geom1[i] == self.block_id and contact.geom2[i] == self.floor_id) or (contact.geom1[i] == self.floor_id and contact.geom2[i] == self.block_id):
                block_pos = self._data.sensor("block_pos").data[:2]
                # dist = max(0., np.linalg.norm(block_pos - self.circle_o) - self.circle_r)
                dist = np.linalg.norm(block_pos - self.circle_o)
                rew += np.exp(-dist)
                # print(rew)
                return rew
        # print(rew)
        return rew
    
    def _compute_terminated(self) -> bool:
        contact = self._data.contact
        for i in range(len(contact.geom1)):
            if (contact.geom1[i] == self.block_id and contact.geom2[i] == self.floor_id) or (contact.geom1[i] == self.floor_id and contact.geom2[i] == self.block_id):
                return True
        return self.time_limit_exceeded()


if __name__ == "__main__":
    env = BasketEnv(render_mode="human")
    env.reset()
    last_obs = None
    errors = []
    for i in range(500):
        # obs, rew, terminated, _, _ = env.step(np.array([0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4]))
        # obs, rew, terminated, _, _ = env.step(np.array([0] * 7))
        action = np.ones(7) * np.pi
        obs, rew, terminated, _, _ = env.step(action)
        # env.reset()
        env.render()
        if last_obs is not None:
            error = last_obs['state']['panda/joint_pos'] + action * env._action_scale - obs['state']['panda/joint_pos']
            errors.append(error)
        last_obs = obs
        # if terminated:
        #     env.reset()
    # for i in range(500):
    #     obs, rew, terminated, _, _ = env.step(np.random.uniform(-3.14, -3.14, 7))
    #     env.render()
    env.close()

    import matplotlib.pyplot as plt
    for i in range(7):
        plt.plot([e[i] for e in errors])
    plt.show()