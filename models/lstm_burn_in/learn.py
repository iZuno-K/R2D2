import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import gym
import baselines.common.tf_util as U
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.deepq.utils import BatchInput, load_state, save_state
from models.lstm_burn_in.build_graph import build_train
from models.lstm_burn_in.replay_buffer import SeqReplayBuffer, PrioritizedReplayBuffer
from queue import Queue


import marlo
import time
from threading import Lock
import csv
from models.lstm_burn_in.utils import BatchMakeThread
from models.lstm_burn_in.utils import reformat_reward

@marlo.threaded
def run_agent(env, sess, coord, buffer, lock, join_token, agent_id, act, zero_state, reward_reformat, kwargs):
    """
        Where agent_id is an integral number starting from 0
        In case, you have requested GPUs, then the agent_id will match
        the GPU device id assigneed to this agent.
    """
    with sess.as_default():
        # env = marlo.init(join_token)
        obs = env.reset()
        done = False
        ep_obs = []
        ep_lstm = []
        lstm_state = zero_state(1)
        ep_action = []
        ep_rew = []
        ep_obs1 = []
        ep_done = []
        while not done:
            ep_obs.append(obs)
            ep_lstm.append(np.squeeze(lstm_state, axis=1))

            # action = env.action_space.sample()
            _action, lstm_state = act(np.array(obs)[None], lstm_state, **kwargs)

            action = _action[0]
            next_obs, reward, done, info = env.step(action)
            reward = reformat_reward(reward) if reward_reformat else reward

            ep_action.append(action)
            ep_rew.append(reward)
            ep_obs1.append(next_obs)
            ep_done.append(float(done))
            obs = next_obs

            # logger.warn("reward(agent{}): {}, action{}".format(agent_id, reward, action))
        with lock:
            buffer.add_path(ep_obs, ep_lstm, ep_action, ep_rew, ep_obs1, ep_done)
        # env.close()

        return ep_rew
        # return ep_obs, ep_lstm, ep_action, ep_rew, ep_obs1, ep_done

def learn(envs,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=10000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          max_length=40,
          burn_in_steps=10,
          lstm_units=128,
          train_iter=1,
          positive_batch_ratio=0.3,
          saved_model=None,
          exploration_initial=1.0,
          reward_reformat=False,
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
        visible_device_list="2",
        # allocator_type='BFC'
        allow_growth=True
    )
)
    sess = tf.Session(config=config)
    # sess.__enter__()

    with sess.as_default():

        env = envs[0]

        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph
        observation_space_shape = env.observation_space.shape
        def make_obs_ph(name):
            return BatchInput(observation_space_shape, name=name)

        def zero_state(batch_size):
            state = np.zeros((2, batch_size, lstm_units))
            return state

        act, train, update_target, debug = build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            lstm_units=lstm_units,
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=param_noise,
            max_length=max_length,
            burn_in_steps=burn_in_steps,
            reward_reformat=reward_reformat
        )

        # Create the replay buffer
        # rb = SeqReplayBuffer(observation_space_shape, env.action_space.n, max_replay_buffer_size=buffer_size, out_seq_length=max_length, burn_in_steps=burn_in_steps, positive_ratio=positive_batch_ratio)
        rb = PrioritizedReplayBuffer(size=buffer_size, alpha=0.9, output_length=max_length, burn_in_length=burn_in_steps)
        # Create the schedule for exploration starting from 1.
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                     # initial_p=1.0,
                                     initial_p=exploration_initial,
                                     final_p=exploration_final_eps)

        # Initialize the parameters and copy them to the target network.
        if saved_model is None:
            U.initialize()
        else:
            print("Load saved model")
            load_state(saved_model)
        update_target()

        episode_rewards = []
        saved_mean_reward = None
        reset = True

        # prepare logger and multi threading
        td = logger.get_dir()
        os.makedirs(td, exist_ok=True)
        f = open(os.path.join(td, "log.csv"), 'w')
        writer = csv.writer(f, lineterminator='\n', delimiter=',')
        writer.writerow(['episode, ret, steps, epsilon'])
        f.flush()
        lock = Lock()
        coord = tf.train.Coordinator()

        episode_rewards = []
        saved_mean_reward = None
        obs = env.reset()
        reset = True
        first = True

        model_saved = False
        model_file = os.path.join(logger.get_dir(), "model")

        episode = 0
        train_t_thresh = learning_starts
        update_target_thresh = learning_starts + 500
        losses = []
        _checkpoint_freq = checkpoint_freq
        _print_freq = print_freq
        t = 0

        batch_queue = Queue(maxsize=train_iter+1)
        batch_maker = BatchMakeThread(batch_queue, rb, batch_size, lock)

        while t < max_timesteps:
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            update_eps = exploration.value(t)
            update_param_noise_threshold = 0.
            kwargs = {'stochastic': True, 'update_eps':update_eps}

            thread_handlers = []
            queues = []
            tmp = time.time()
            print("start working")
            for env in envs:
                # Run agent-N on a separate thread
                thread_handler, queue = run_agent(env, sess, coord, rb, lock, 0, 0, act, zero_state, reward_reformat, kwargs)
                # Accumulate thread handlers
                thread_handlers.append(thread_handler)
                queues.append(queue)
            # Wait for  threads to complete or raise an exception
            coord.join(thread_handlers)
            [th.join() for th in thread_handlers]
            print("worked time", time.time() - tmp)
            sum_returns = 0.
            sum_lengths = 0
            valid_ep = 0

            for q in queues:
                ep_rew = q.get()
                # if type(ep_rew) == tuple:
                if type(ep_rew) == list:
                    # rb.add_path(*ep)
                    # sum_returns += sum(ep[3])
                    # sum_lengths += len(ep[3])
                    # t += len(ep[3])
                    sum_returns += sum(ep_rew)
                    t += len(ep_rew)
                    episode += 1
                    valid_ep += 1
                    episode_rewards.append(sum(ep_rew))
            if valid_ep > 0:
                averaged_return = sum_returns / valid_ep
                writer.writerow([episode, averaged_return, t, update_eps])
                f.flush()
            print("step:{}, episode:{}, rb_size:{}".format(t, episode, len(rb)))

            # if t > learning_starts and t > train_t_thresh :

            if t > learning_starts:
                if first:
                    batch_maker.start()
                    first = False
                for k in range(train_iter):
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    start = time.time()
                    # print("batch sampling")
                    obs_seq, batch_lstm, action_seq, reward_seq, obs1_seq, done_seq, lengths, burn_in_seq, burn_in_lengths, weights, batch_idxes, size_positive = batch_queue.get()
                    # obs_seq, batch_lstm, action_seq, reward_seq, obs1_seq, done_seq, lengths, burn_in_seq, burn_in_lengths, weights, batch_idxes = rb.sample(batch_size, beta=prioritized_replay_beta0)
                    print("batch sample: {}s".format(time.time() - start))

                    # start = time.time()
                    obs_seq = np.array(obs_seq).reshape((-1,) + observation_space_shape)  # (batch_size*max_length, obs_shape)
                    action_seq = np.array(action_seq).reshape(-1)
                    reward_seq = np.array(reward_seq).reshape(-1)
                    obs1_seq = np.array(obs1_seq).reshape((-1,) + observation_space_shape)  # (batch_size*max_length, obs_shape)
                    done_seq = np.array(done_seq).reshape(-1)
                    burn_in_seq = np.array(burn_in_seq).reshape((-1,) + observation_space_shape)
                    batch_lstm = np.array(batch_lstm).transpose(1, 0, 2)
                    # print("reshape: {}s".format(time.time() - start))

                    # start = time.time()
                    td_error, loss = train(obs_seq, batch_lstm, action_seq, reward_seq, obs1_seq, done_seq, lengths, burn_in_seq, burn_in_lengths)
                    # print("loss: {}s".format(time.time() - start))
                    losses.append(loss)
                    if prioritized_replay:
                        # start = time.time()
                        # calc priority
                        eta = 0.9
                        priority = np.abs(td_error).reshape(-1, max_length)  # (batch, max_length)
                        new_priorities = eta * np.max(priority, axis=1) + (1 - eta) * np.sum(priority, axis=1) / np.asarray(lengths)  # (batch)
                        rb.update_priorities(batch_idxes, new_priorities)
                        # print("update priority: {}s".format(time.time() - start))

                # train_t_thresh = t +100

            if t > learning_starts and t > update_target_thresh:
                # Update target network periodically.
                update_target()
                update_target_thresh = t + 2500

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if print_freq is not None and t > _print_freq:
                # eval
                eval_done = False
                eval_obs = env.reset()
                eval_lstm = zero_state(1)
                eval_kwargs = {'stochastic': False, 'update_eps': -1}
                eval_ret = 0.
                while not eval_done:
                    eval_act, eval_lstm = act(np.array(obs)[None], eval_lstm, **eval_kwargs)
                    eval_obs, eval_rew, eval_done, _ = env.step(eval_act[0])
                    eval_rew = reformat_reward(eval_rew) if reward_reformat else eval_rew
                    eval_ret += eval_rew

                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.record_tabular("mean 100 episode loss", np.mean(losses))
                logger.record_tabular("eval reward", eval_ret)
                logger.dump_tabular()

                _print_freq = t + print_freq


            if checkpoint_freq is not None and t > learning_starts and num_episodes > 100 and t > _checkpoint_freq:
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
                    _checkpoint_freq = t + checkpoint_freq
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_state(model_file)

        batch_maker.stop()

    return act
