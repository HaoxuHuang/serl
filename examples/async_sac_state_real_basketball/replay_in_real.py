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


import threading
import pynput

EMERGENCY_FLAG = threading.Event()
EMERGENCY_FLAG.clear()  # clear() to continue the actor/learner loop, set() to pause

def pause_callback(key):
    """Callback for when a key is pressed"""
    global EMERGENCY_FLAG
    try:
        # chosen a rarely used key to avoid conflicts. this listener is always on, even when the program is not in focus
        if key == pynput.keyboard.Key.f1:
            print("Emergency.")
            # set the PAUSE FLAG to pause the actor/learner loop
            EMERGENCY_FLAG.set()
    except AttributeError:
        # print(f'{key} pressed')
        pass


listener = pynput.keyboard.Listener(
    on_press=pause_callback
)  # to enable keyboard based pause
listener.start()


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

    history = []

    with open(args.demo_path, 'rb') as f:
        demo_data = pickle.load(f)
    for it in demo_data:
        # pos = it['observations'][:7]
        # env.set_joint_pos(pos)
        action = it['actions']
        aaa = env.step(action)
        obs, rew, terminated, _, _ = aaa
        history.append({'demo': it,'real': aaa})

        if EMERGENCY_FLAG.is_set():
            break_flag = False
            while True:
                response = input(
                    "Do you want to continue (c) or exit (e)? "
                )
                if response == "c" :
                    EMERGENCY_FLAG.clear()
                    print("Continuing")
                    break
                elif response == 'e':
                    print("Stopping")
                    break_flag = True
                    break
                else:
                    print('Unknown input')
            if break_flag:
                break

        if it['dones']:
            print("Episode done, resetting environment.")
            env.reset()
    
    with open('demo_vs_real.pkl','wb') as f:
        pickle.dump(history,f)
    env.close()
