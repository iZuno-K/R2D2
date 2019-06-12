import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import gym
import models.lstm_burn_in.baselines_utils as U
from models.lstm_burn_in.baselines_utils import LinearSchedule
from models.lstm_burn_in.baselines_utils import BatchInput, load_state, save_state
from models.lstm_burn_in.build_graph import build_train
from models.lstm_burn_in.replay_buffer import PrioritizedReplayBuffer, PositiveReplayBuffer
from models.lstm_burn_in.utils import env_detector

import marlo
import cv2
from models.lstm_burn_in import baselines_logger as logger_b
from threading import Thread, Lock
import csv
import time
from models.lstm_burn_in.utils import BatchMakeThread, reformat_reward, reformat_obs, Worker
from sys import getsizeof
from queue import Queue
from marlo.base_env_builder import logger
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0
logger.level=WARN

def act_for_mobchase(act_func, env_name):
    if 'Mobchase' in env_name:
        print('make act func for Mobchase')
        def _act(ob, _lstm_state, stochastic=True, update_eps=-1):
            a = act_func(ob, _lstm_state, stochastic=stochastic, update_eps=update_eps)
            modified_action = [7] if a[0][0] == 5 else a[0]
            a.append(modified_action)
            return a  # [action, lstm, action_into_env]
    elif 'Apartment' in env_name:
        def _act(ob, _lstm_state, stochastic=True, update_eps=-1):
            a = act_func(ob, _lstm_state, stochastic=stochastic, update_eps=update_eps)
            modified_action = [a[0][0] + 1]
            a.append(modified_action)
            return a  # [action, lstm, action_into_env]
    elif 'Cliff' in env_name:
        def _act(ob, _lstm_state, stochastic=True, update_eps=-1):
            a = act_func(ob, _lstm_state, stochastic=stochastic, update_eps=update_eps)
            modified_action = [a[0][0] + 1]
            a.append(modified_action)
            return a  # [action, lstm, action_into_env]
    else:
        def _act(ob, _lstm_state, stochastic=True, update_eps=-1):
            a = act_func(ob, _lstm_state, stochastic=stochastic, update_eps=update_eps)
            modified_action = a[0]
            a.append(modified_action)
            return a  # [action, lstm, action_into_env]

    return _act


def get_join_tokens(env_params):
    print("Generating join tokens locally...")
    total_join_tokens = []
    for param in env_params:
        env_name = param[0]
        client_pool = param[1]
        join_tokens = marlo.make(env_name,
                                 params={
                                     "client_pool": client_pool,
                                 })
        total_join_tokens += join_tokens
    # print("join tokens len: {}".format(len(join_tokens)))
    return total_join_tokens


def learn(
          env_params,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=500,
          checkpoint_freq=1000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          max_length=80,
          burn_in_steps=40,
          lstm_units=128,
          train_iter=1,
          initial_positive_batch_ratio=0.5,
          ip='127.0.0.1',
          saved_model=None,
          exploration_initial=1.0,
          reward_reformat=True,
          ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="3",
            allocator_type='BFC',
            # allow_growth=True
        )
    )
    # config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)

    with sess.as_default():
        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph
        observation_space_shape = (90, 120, 3)
        def make_obs_ph(name):
            return BatchInput(observation_space_shape, name=name)

        def zero_state(batch_size):
            state = np.zeros((2, batch_size, lstm_units))
            return state

        # marlo set up ----------------------------------------------------
        join_tokens = get_join_tokens(env_params)
        envs = [marlo.init(token) for token in join_tokens]
        if not join_tokens:
            return
        _idxes = np.arange(len(join_tokens))
        lock = Lock()
        # env_name = env_params[0][0]
        env_name = env_detector(envs[0])

        print(envs[0].action_names)

        if 'Mobchase' in env_name:
            num_action = 6
        elif 'Apartment' in env_name:
            num_action = 6
        elif 'Cliff' in env_name:
            num_action = 4
        else:
            num_action = len(envs[0].action_names[0])

        _act_f, worker_act_fs, update_worker_fs, train, update_target, debug = build_train(
            worker_num=len(env_params),
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=num_action,  # ignore 5, 6 but use 7
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            lstm_units=lstm_units,
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=param_noise,
            max_length=max_length,
            burn_in_steps=burn_in_steps,
            reward_reformat=reward_reformat
        )
        worker_act_fs = [act_for_mobchase(act, env_name) for act in worker_act_fs]

        # Create the replay buffer
        # rb = PrioritizedReplayBuffer(size=buffer_size, alpha=0.9, output_length=max_length, burn_in_length=burn_in_steps)
        rb0 = PrioritizedReplayBuffer(size=buffer_size, alpha=0.9, output_length=max_length, burn_in_length=burn_in_steps)
        rb1 = PrioritizedReplayBuffer(size=buffer_size, alpha=0.9, output_length=max_length, burn_in_length=burn_in_steps)
        rb = PositiveReplayBuffer(rb0, rb1)
        # Create the schedule for exploration starting from 1.
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)
        positive_ratio = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                        initial_p=initial_positive_batch_ratio,
                                        final_p=0.1)

        # Initialize the parameters and copy them to the target network.
        coord = tf.train.Coordinator()
        if saved_model is None:
            U.initialize()
        else:
            print("Load saved model")
            load_state(saved_model)
        update_target()
        # print("main: {}".format(tf.get_default_session()))

        # logging set up
        episode_rewards = []
        saved_mean_reward = None
        reset = True

        td = logger_b.get_dir()
        model_saved = False
        model_file = os.path.join(td, "model")

        # training -------------------------------------------------------
        episode = 0
        train_t_thresh = learning_starts
        update_target_thresh = learning_starts + 500
        t = 0
        losses = []
        _checkpoint_freq = checkpoint_freq
        _print_freq = print_freq

        first = True

        episode_step_queue = Queue(maxsize=1)
        episode_step_queue.put([0, 0])
        return_queue = Queue()

        # set_worker
        workers = []
        for i in range(len(env_params)):
            _envs = [envs[i]]
            env_name = env_params[i][0]
            _join_tokens = [join_tokens[i]]
            _agent_ids = [i]
            # sess
            # coord
            _act = worker_act_fs[i]
            _update = update_worker_fs[i]
            # zero_state
            # reward_reformat
            # exploration
            # rb
            # lock
            # episode_step_queue
            workers.append(Worker(_envs, env_name, _join_tokens, _agent_ids, sess, coord, _act, _update,
                                  zero_state, reward_reformat, exploration, rb, lock, episode_step_queue,
                                  return_queue))

        batch_queue = Queue(maxsize=train_iter + 1)
        batch_maker = BatchMakeThread(batch_queue, rb, batch_size, lock)
        batch_started = False
        episode_rewards =[]
        print("learning start")
        train_num = 0

        while t < max_timesteps:
            if callback is not None:
                if callback(locals(), globals()):
                    break

            # get total step information
            ep_step = episode_step_queue.get()
            episode = ep_step[0]
            t = ep_step[1]
            episode_step_queue.put(ep_step)
            if t < learning_starts:
                time.sleep(1)

            # Take action and update exploration to the newest value
            if hasattr(rb, 'update_ratio'):
                rb.update_ratio(positive_ratio.value(t))
            update_param_noise_threshold = 0.

            # optimization
            if len(rb) > batch_size and not batch_started:
                batch_maker.start()
                batch_started = True
            if t > learning_starts:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                with lock:
                    obs_seq, batch_lstm, action_seq, reward_seq, obs1_seq, done_seq, lengths, burn_in_seq, burn_in_lengths, weights, batch_idxes, pos_size = batch_queue.get()
                    # obs_seq, batch_lstm, action_seq, reward_seq, obs1_seq, done_seq, lengths, burn_in_seq, burn_in_lengths, weights, batch_idxes = batch_queue.get()
                    # obs_seq, batch_lstm, action_seq, reward_seq, obs1_seq, done_seq, lengths, burn_in_seq, burn_in_lengths, weights, batch_idxes = rb.sample(batch_size, beta=prioritized_replay_beta0)

                obs_seq = np.array(obs_seq).reshape((-1,) + observation_space_shape)  # (batch_size*max_length, obs_shape)
                action_seq = np.array(action_seq).reshape(-1)
                reward_seq = np.array(reward_seq).reshape(-1)
                obs1_seq = np.array(obs1_seq).reshape((-1,) + observation_space_shape)  # (batch_size*max_length, obs_shape)
                done_seq = np.array(done_seq).reshape(-1)
                burn_in_seq = np.array(burn_in_seq).reshape((-1,) + observation_space_shape)
                batch_lstm = np.array(batch_lstm).transpose(1, 0, 2)

                td_errors, loss = train(obs_seq, batch_lstm, action_seq, reward_seq, obs1_seq, done_seq, lengths, burn_in_seq, burn_in_lengths)
                # coord.join()
                losses.append(loss)
                train_num += 1

                if prioritized_replay:
                    # start = time.time()
                    # calc priority
                    eta = 0.9
                    priority = np.abs(td_errors).reshape(-1, max_length)  # (batch, max_length)
                    new_priorities = eta * np.max(priority, axis=1) + (1 - eta) * np.sum(priority, axis=1) / np.asarray(
                        lengths)  # (batch)
                    with lock:
                        rb.update_priorities(batch_idxes, pos_size, new_priorities)
                        # rb.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t > update_target_thresh:
                # Update target network periodically.
                update_target()
                update_target_thresh = t + 2500

            while not return_queue.empty():
                episode_rewards.append(return_queue.get())
            if len(episode_rewards) > 0:
                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)

            if print_freq is not None and t > _print_freq:
                # if first:
                #     tmp = time.time()
                #     tmp_obs = np.array(rb._storage[0][0][0])
                #     tmp_kwg = {'stochastic': False, 'update_eps': -1}
                #     _ = act(np.array(tmp_obs)[None], zero_state(1), **kwargs)
                #     tmp_obs = None
                #     tmp_kwg = None
                #     _ = None
                #     print("first init time ({}s)".format(time.time() - tmp))
                #     first = False

                # evaluation (evaluation data is also appended in replay buffer)
                # eval_length = 1
                # eval_kwargs = {'stochastic': False, 'update_eps': -1}
                # eval_returns = []
                # for eval in range(eval_length):
                #     start = time.time()
                #     print("eval episode:{}".format(eval))
                #
                #     thread_handlers = []
                #     queues = []
                #     for _idx, env, join_token in zip(_idxes, envs, join_tokens):
                #         thread_handler, queue = run_agent(env, sess, coord, rb, lock, join_token, _idx, act, zero_state, reward_reformat, eval_kwargs)
                #         thread_handlers.append(thread_handler)
                #         queues.append(queue)
                #     coord.join(thread_handlers)
                #     [th.join() for th in thread_handlers]
                #
                #     eval_sum_rew = 0.
                #     for q in queues:
                #         ep_rew = q.get()
                #         if type(ep_rew) == list:
                #             eval_sum_rew += sum(ep_rew)
                #             t += len(ep_rew)
                #             valid_ep += 1
                #             eval_returns.append(sum(ep_rew))
                #     episode += 1
                #     print("eval episode:{} done ({} s)".format(eval, time.time() - start))

                logger_b.record_tabular("steps", t)
                logger_b.record_tabular("episodes", episode)
                logger_b.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger_b.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                # logger_b.record_tabular("mean eval reward", sum(eval_returns) / eval_length)
                logger_b.record_tabular("mean 100 episode loss", np.mean(losses))
                logger_b.record_tabular("optimization iteration", train_num)
                logger_b.record_tabular("buffer size", len(rb))
                if hasattr(rb, 'rb_positive'):
                    logger_b.record_tabular("positive buffer size", len(rb.rb_positive))
                logger_b.dump_tabular()

                _print_freq = t + print_freq

            if checkpoint_freq is not None and t > learning_starts and episode > 100  and t > _checkpoint_freq:
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger_b.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
                    # coord.join()
                    _checkpoint_freq = t + checkpoint_freq

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_state(model_file)

        for worker in workers:
            worker.stop()

        return
