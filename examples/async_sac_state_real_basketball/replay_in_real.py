import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
import pickle
import argparse

import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    HandGuidance
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
import jax


def parse_args():
    parser = argparse.ArgumentParser(
        description="Basketball Replay in Real Environment",)
    parser.add_argument(
        "--demo_path",
        type=str,
        default=None,
        help="Path to the demo data file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    env = gym.make("FrankaBasketball-State-v0", save_video=False)
    env = HandGuidance(env)
    env = SERLObsWrapper(env)
    image_keys = [k for k in env.observation_space.keys() if "state" not in k]

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    obs, _ = env.reset()

    args = parse_args()
    if args.demo_path is None:
        raise ValueError("Please provide a demo path using --demo_path")

    with open(args.demo_path, 'rb') as f:
        demo_data = pickle.load(f)
        for it in demo_data:
            # pos = it['observations'][:7]
            # env.set_joint_pos(pos)
            action = it['actions']
            obs, rew, terminated, _, _ = env.step(action)

            env.render()
            # if terminated:
            #     env.reset
            if it['dones']:
                print("Episode done, resetting environment.")
                env.reset()
    env.close()
