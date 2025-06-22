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

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("port", 5498, "Port number")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("load_checkpoint", None, "Path to load checkpoints.")

flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step"
)
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_integer("utd_ratio", 1, "utd_ratio.")
flags.DEFINE_integer("sleep_time", 0, "Sleep time.")

flags.DEFINE_float("action_scale", 0.2, "Scale applied to agent actions.")
flags.DEFINE_float("angle_penalty", 1e-5, "Penalty coefficient for joint angles.")
flags.DEFINE_float("energy_penalty", 1e-2, "Penalty coefficient for energy usage.")
# flags.DEFINE_integer("seed", 0, "Random seed for environment/simulation.")
flags.DEFINE_float("control_dt", 0.02, "Control timestep (seconds).")
flags.DEFINE_float("physics_dt", 0.002, "Physics simulation timestep (seconds).")
flags.DEFINE_float("time_limit", 10.0, "Maximum episode duration (seconds).")

flags.DEFINE_float("discount", 0.999, "Discount.")

flags.DEFINE_string("data_store_path", None, "Path to save and load data_store.")
flags.DEFINE_boolean("teacher", False, "Is this a teacher agent.")
flags.DEFINE_boolean("load_offline_data", False, "Load offline data.")
flags.DEFINE_integer("offline_decay_start", None, "Offline decay start step.")
flags.DEFINE_integer("offline_decay_steps", None, "Offline decay steps.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline decay ratio.")

flags.DEFINE_integer("teacher_episodes", 5, "Actor episodes")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(rng, agent: SACAgent, demo, checkpoints_path, sampling_rng):
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

    modes = [[] for _ in range(7)]
    stds = [[] for _ in range(7)]

    for update_steps, checkpoint_path in checkpoints_path:
        params = checkpoints.restore_checkpoint(checkpoint_path, target=None)
        params = params["params"]
        agent = agent.replace(state=agent.state.replace(params=params))
        print_green("Loaded checkpoint from " + checkpoint_path)

        with timer.context("train"):
            obs = []
            for transition in demo:
                obs.append(transition["observations"])
            obs = jnp.array(obs)
            obs = jax.device_put(
                obs, jax.sharding.PositionalSharding(jax.local_devices())
            )
            sampling_rng, key = jax.random.split(sampling_rng)
            dist = agent.forward_policy(obs, rng=key, train=False)
            std = dist.distribution._scale_diag
            # mode = dist.mode()
            from serl_launcher.networks.actor_critic_nets import (
                TanhMultivariateNormalDiag,
            )

            dist = TanhMultivariateNormalDiag(
                dist.distribution._loc, dist.distribution._scale_diag * 0.3
            )
            mode = dist.sample(seed=sampling_rng)
            mode = np.asarray(jax.device_get(mode))
            std = np.asarray(jax.device_get(std))
            trajectory_mode = [[] for _ in range(7)]
            trajectory_std = [[] for _ in range(7)]
            for i in range(7):
                for j in range(len(obs)):
                    trajectory_mode[i].append(mode[j][i])
                    trajectory_std[i].append(std[j][i])

            # sampling_rng, key = jax.random.split(sampling_rng)
            # dist = agent.forward_policy(obs, rng=key, train=False)
            # mode = dist.mode()
            # std = dist.distribution._scale_diag
            # mode = np.asarray(jax.device_get(mode))
            # std = np.asarray(jax.device_get(std))
            # for i in range(7):
            #     trajectory_mode[i].append(mode[i])
            #     trajectory_std[i].append(std[i])

        for i in range(7):
            modes[i].append(trajectory_mode[i])
            stds[i].append(trajectory_std[i])

        # if wandb_logger:
        #     wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)
        #     wandb_logger.log(info, step=update_steps)

    for i in range(7):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        correct_actions = [transition["actions"][i] for transition in demo]

        plt.plot(correct_actions, label="Correct Actions", linestyle="--", color="red")
        for idx, trajectory in enumerate(modes[i]):
            print(len(trajectory))
            plt.plot(trajectory, label=f"Update Step {checkpoints_path[idx][0]}")
        
        # concatenated_trajectory = []
        # colors = plt.cm.viridis(np.linspace(0, 1, len(modes[i])))
        # concatenated_trajectory.extend(correct_actions)
        # plt.plot(
        #     range(
        #         len(concatenated_trajectory) - len(correct_actions),
        #         len(concatenated_trajectory),
        #     ),
        #     correct_actions,
        #     label="Correct Actions",
        #     linestyle="--",
        #     color="grey",
        # )
        # for idx, trajectory in enumerate(modes[i]):
        #     concatenated_trajectory.extend(trajectory)
        #     plt.plot(
        #         range(
        #             len(concatenated_trajectory) - len(trajectory),
        #             len(concatenated_trajectory),
        #         ),
        #         trajectory,
        #         label=f"Update Step {checkpoints_path[idx][0]}",
        #         color=colors[idx],
        #     )
        #     plt.plot(
        #         range(
        #             len(concatenated_trajectory) - len(correct_actions),
        #             len(concatenated_trajectory),
        #         ),
        #         correct_actions,
        #         linestyle="--",
        #         color="grey",
        #     )
        
        plt.title(f"Trajectory Modes for Joint {i}")
        plt.xlabel("Time Step")
        plt.ylabel("Mode Value")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))

        for idx, trajectory in enumerate(stds[i]):
            plt.plot(trajectory, label=f"Update Step {checkpoints_path[idx][0]}")

        # concatenated_trajectory = []
        # colors = plt.cm.viridis(np.linspace(0, 1, len(stds[i])))
        # for idx, trajectory in enumerate(stds[i]):
        #     concatenated_trajectory.extend(trajectory)
        #     plt.plot(
        #         range(
        #             len(concatenated_trajectory) - len(trajectory),
        #             len(concatenated_trajectory),
        #         ),
        #         trajectory,
        #         label=f"Update Step {checkpoints_path[idx][0]}",
        #         color=colors[idx],
        #     )

        plt.title(f"Trajectory Standard Deviations for Joint {i}")
        plt.xlabel("Time Step")
        plt.ylabel("Standard Deviation Value")
        plt.legend()
        plt.grid()
        plt.show()

    print("Actory loop finished")


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

    import pickle as pkl

    with open(FLAGS.data_store_path, "rb") as f:
        demo = pkl.load(f)

    checkpoints_path = []

    root_path = "/home/drl/Code/serl/examples/async_sac_state_real_basketball/checkpoints/checkpoints_2025-06-21_17-22-08"
    steps = range(5000, 25001, 5000)
    for step in steps:
        checkpoint_path = os.path.join(root_path, f"checkpoint_{step}")
        checkpoints_path.append((step, checkpoint_path))

    actor(
        sampling_rng,
        agent,
        demo,
        checkpoints_path,
        sampling_rng,
    )


if __name__ == "__main__":
    app.run(main)
