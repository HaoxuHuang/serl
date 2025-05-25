#!/usr/bin/env python3

from basketball_env_monitor import BasketMonitorWrapper
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


import basketball_environment
register(
    id="Basket-v0",
    entry_point="basketball_environment:BasketEnv",
    max_episode_steps=1000,
)

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
flags.DEFINE_integer("port", 5498, "Port number")
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

flags.DEFINE_string("data_store_path", None, "Path to save and load data_store.")
flags.DEFINE_boolean("teacher", False, "Is this a teacher agent.")
flags.DEFINE_boolean("load_offline_data", False, "Load offline data.")
flags.DEFINE_integer("offline_data_steps", None, "Number of offline data steps.")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent: SACAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """

    if not FLAGS.teacher:
        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            make_trainer_config(FLAGS.port, FLAGS.port + 1),
            data_store,
            wait_for_server=not FLAGS.teacher,
        )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    if not FLAGS.teacher:
        client.recv_network_callback(update_params)

    if FLAGS.teacher and FLAGS.load_checkpoint != None:
        params = checkpoints.restore_checkpoint(FLAGS.load_checkpoint, target=None)
        params = params["params"]
        agent = agent.replace(state=agent.state.replace(params=params))
        print_green('Loaded checkpoint from ' + FLAGS.load_checkpoint)

    eval_env = gym.make(FLAGS.env,
                        action_scale=FLAGS.action_scale,
                        angle_penalty=FLAGS.angle_penalty,
                        energy_penalty=FLAGS.energy_penalty,
                        seed=FLAGS.seed,
                        control_dt=FLAGS.control_dt,
                        physics_dt=FLAGS.physics_dt,
                        time_limit=FLAGS.time_limit)
    if FLAGS.env == "PandaPickCube-v0" or FLAGS.env == "Basket-v0":
        eval_env = gym.wrappers.FlattenObservation(eval_env)
    eval_env = RecordEpisodeStatistics(eval_env)

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    full_history = []
    for step in tqdm.tqdm(range(FLAGS.actor_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
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

            new_data = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done or truncated,
            )

            data_store.insert(new_data)

            if FLAGS.teacher and FLAGS.data_store_path is not None:
                full_history.append(new_data)

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

        if step % FLAGS.steps_per_update == 0:
            if not FLAGS.teacher:
                client.update()

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            if not FLAGS.teacher:
                client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            if not FLAGS.teacher:
                client.request("send-stats", stats)

    if FLAGS.teacher and FLAGS.data_store_path is not None:
        import os
        import pickle
        os.makedirs(FLAGS.data_store_path, exist_ok=True)
        with open(os.path.join(FLAGS.data_store_path, 'data_store.pkl'), "wb") as f:
            pickle.dump(full_history, f)
        print_green('Saved data store to ' + FLAGS.data_store_path)


##############################################################################


def learner(rng, agent: SACAgent, replay_buffer, replay_iterator, offline_data, offline_iterator):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    if FLAGS.load_checkpoint != None:
        params = checkpoints.restore_checkpoint(FLAGS.load_checkpoint, target=None)
        params = params["params"]
        agent = agent.replace(state=agent.state.replace(params=params))
        print_green('Loaded checkpoint from ' + FLAGS.load_checkpoint)

    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(FLAGS.port, FLAGS.port + 1),
                           request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    if offline_data is None:
        pbar = tqdm.tqdm(
            total=FLAGS.training_starts,
            initial=len(replay_buffer),
            desc="Filling up replay buffer",
            position=0,
            leave=True,
        )
        while len(replay_buffer) < FLAGS.training_starts:
            pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
            time.sleep(1)
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.learner_steps), dynamic_ncols=True, desc="learner"):
        # Train the networks
        with timer.context("sample_replay_buffer"):
            batch = next(replay_iterator)
        
        if offline_data is not None and (FLAGS.offline_data_steps is None or update_steps < FLAGS.offline_data_steps):
            with timer.context("sample_offline_data"):
                offline_batch = next(offline_iterator)
                ratio = 0.5
                batch = jax.tree_map(
                    lambda x, y: jnp.concatenate(
                        [
                            x[: int(x.shape[0] * ratio)],
                            y[: int(y.shape[0] * (1 - ratio))],
                        ],
                        axis=0,
                    ),
                    batch,
                    offline_batch,
                )

        with timer.context("train"):
            agent, update_info = agent.update_high_utd(
                batch, utd_ratio=FLAGS.utd_ratio)
            agent = jax.block_until_ready(agent)

            # publish the updated network
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log(
                {"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=100
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


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

    if FLAGS.learner:
        sampling_rng = jax.device_put(
            sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type="replay_buffer",
            preload_rlds_path=FLAGS.preload_rlds_path,
        )
        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size * FLAGS.critic_actor_ratio,
            },
            device=sharding.replicate(),
        )
        if FLAGS.load_offline_data and FLAGS.data_store_path is not None:
            import os
            from serl_launcher.data.data_store import populate_data_store
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
            offline_data = populate_data_store(
                offline_data, [os.path.join(FLAGS.data_store_path, 'data_store.pkl')])
            replay_buffer = populate_data_store(
                replay_buffer, [os.path.join(FLAGS.data_store_path, 'data_store.pkl')])
            print_green('Loaded offline data store from ' + FLAGS.data_store_path)
        else:
            offline_data = None
            offline_iterator = None
        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            replay_iterator,
            offline_data,
            offline_iterator
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
