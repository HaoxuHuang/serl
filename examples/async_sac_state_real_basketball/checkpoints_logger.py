#!/usr/bin/env python3

import time
from functools import partial

import os
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.utils.launcher import (
    make_sac_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)

from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.data.data_store import populate_data_store

import franka_sim

from gym.envs.registration import register


import basketball_sim_environment

register(
    id="Basket-v0",
    entry_point="basketball_sim_environment:BasketEnv",
    max_episode_steps=1000,
)
from basketball_env_monitor import BasketMonitorWrapper

FLAGS = flags.FLAGS

# flags.DEFINE_string("env", "HalfCheetah-v4", "Name of environment.")
# flags.DEFINE_string("env", "PandaPickCube-v0", "Name of environment.")
flags.DEFINE_string("env", "Basket-v0", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 8, "critic to actor update ratio.")

flags.DEFINE_integer("actor_steps", 1000000, "Maximum number of actor steps.")
flags.DEFINE_integer("learner_steps", 1000000, "Maximum number of learner steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("port", 5488, "Port number")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("load_checkpoint", None, "Path to load checkpoints.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_integer("utd_ratio", 1, "utd_ratio.")
flags.DEFINE_integer("sleep_time", 0, "Sleep time.")

flags.DEFINE_float("action_scale", 0.2, "Scale applied to agent actions.")
flags.DEFINE_float("angle_penalty", 1e-5, "Penalty coefficient for joint angles.")
flags.DEFINE_float("energy_penalty", 1e-4, "Penalty coefficient for energy usage.")
# flags.DEFINE_integer("seed", 0, "Random seed for environment/simulation.")
flags.DEFINE_float("control_dt", 0.02, "Control timestep (seconds).")
flags.DEFINE_float("physics_dt", 0.002, "Physics simulation timestep (seconds).")
flags.DEFINE_float("time_limit", 10.0, "Maximum episode duration (seconds).")

flags.DEFINE_float("discount", 0.9999, "Discount.")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def learner(rng, agent: SACAgent, offline_data, offline_iterator, checkpoints_path):
    """
    The learner loop, which runs when "--learner" is set to True.
    """

    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    fixed_offline_batch = next(offline_iterator)
    import copy

    fixed_offline_batch = copy.deepcopy(fixed_offline_batch)

    for update_steps, checkpoint_path in checkpoints_path:
        params = checkpoints.restore_checkpoint(checkpoint_path, target=None)
        params = params["params"]
        agent = agent.replace(state=agent.state.replace(params=params))
        print_green("Loaded checkpoint from " + checkpoint_path)

        with timer.context("train"):
            q_info = agent.get_q_info(fixed_offline_batch, utd_ratio=FLAGS.utd_ratio)
            agent = jax.block_until_ready(agent)
            q_info = {k: float(v.item()) for k, v in q_info.items()}
            print(q_info)

        if wandb_logger:
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)
            wandb_logger.log(q_info, step=update_steps)

    print("Learner loop finished")


##############################################################################


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    if FLAGS.render:
        env = gym.make(
            FLAGS.env,
            render_mode="human",
            action_scale=FLAGS.action_scale,
            angle_penalty=FLAGS.angle_penalty,
            energy_penalty=FLAGS.energy_penalty,
            seed=FLAGS.seed,
            control_dt=FLAGS.control_dt,
            physics_dt=FLAGS.physics_dt,
            time_limit=FLAGS.time_limit,
        )

    else:
        env = gym.make(
            FLAGS.env,
            action_scale=FLAGS.action_scale,
            angle_penalty=FLAGS.angle_penalty,
            energy_penalty=FLAGS.energy_penalty,
            seed=FLAGS.seed,
            control_dt=FLAGS.control_dt,
            physics_dt=FLAGS.physics_dt,
            time_limit=FLAGS.time_limit,
        )

    if FLAGS.env == "PandaPickCube-v0" or FLAGS.env == "Basket-v0":
        env = gym.wrappers.FlattenObservation(env)

    if FLAGS.debug:
        env = BasketMonitorWrapper(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_agent(
        seed=FLAGS.seed,
        discount=FLAGS.discount,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: SACAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    sampling_rng = jax.device_put(sampling_rng, sharding.replicate())

    offline_data = make_replay_buffer(
        env,
        capacity=FLAGS.replay_buffer_capacity,
        rlds_logger_path=FLAGS.log_rlds_path,
        type="replay_buffer",
        preload_rlds_path=FLAGS.preload_rlds_path,
    )
    offline_iterator = offline_data.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size * FLAGS.critic_actor_ratio,
        },
        device=sharding.replicate(),
    )
    offline_data = populate_data_store(offline_data, [FLAGS.data_store_path])

    checkpoints_path = []

    root_path = ""
    steps = []
    for step in steps:
        checkpoint_path = os.path.join(root_path, f"checkpoint_{step}")
        checkpoints_path.append((step, checkpoint_path))

    learner(
        sampling_rng,
        agent,
        offline_data,
        offline_iterator,
        checkpoints_path,
    )


if __name__ == "__main__":
    app.run(main)
