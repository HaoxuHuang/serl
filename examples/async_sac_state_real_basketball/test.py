#!/usr/bin/env python3

import time
from functools import partial

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
flags.DEFINE_string(
    "exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 8, "critic to actor update ratio.")

flags.DEFINE_integer("actor_steps", 1000000, "Maximum number of actor steps.")
flags.DEFINE_integer("learner_steps", 1000000, "Maximum number of learner steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000,
                     "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300,
                     "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300,
                     "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30,
                     "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer(
    "eval_n_trajs", 5, "Number of trajectories for evaluation.")

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


def actor(agent: SACAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """

    assert FLAGS.load_checkpoint != None
    params = checkpoints.restore_checkpoint(FLAGS.load_checkpoint, target=None)
    params = params["params"]
    agent = agent.replace(state=agent.state.replace(params=params))

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    for step in tqdm.tqdm(range(FLAGS.actor_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                # argmax=True,
                seed=key,
                deterministic=False,
            )
            actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)

            running_return += reward

            data_store.insert(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done or truncated,
                )
            )

            obs = next_obs
            if done or truncated:
                running_return = 0.0
                if FLAGS.debug:
                    env.plot_rewards()
                    env.plot_observation_component()
                    env.plot_actions()
                else:
                    time.sleep(FLAGS.sleep_time)
                obs, _ = env.reset()

        if FLAGS.render:
            env.render()

        timer.tock("total")


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    if FLAGS.render:
        env = gym.make(FLAGS.env,
                       render_mode="human",
                       action_scale=FLAGS.action_scale,
                       angle_penalty=FLAGS.angle_penalty,
                       energy_penalty=FLAGS.energy_penalty,
                       seed=FLAGS.seed,
                       control_dt=FLAGS.control_dt,
                       physics_dt=FLAGS.physics_dt,
                       time_limit=FLAGS.time_limit)

    else:
        env = gym.make(FLAGS.env,
                       action_scale=FLAGS.action_scale,
                       angle_penalty=FLAGS.angle_penalty,
                       energy_penalty=FLAGS.energy_penalty,
                       seed=FLAGS.seed,
                       control_dt=FLAGS.control_dt,
                       physics_dt=FLAGS.physics_dt,
                       time_limit=FLAGS.time_limit)

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
    data_store = QueuedDataStore(2000)  # the queue size on the actor

    # actor loop
    print_green("starting actor loop")
    actor(agent, data_store, env, sampling_rng)


if __name__ == "__main__":
    app.run(main)
