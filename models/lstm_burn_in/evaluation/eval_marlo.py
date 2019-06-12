import os
import tensorflow as tf
import numpy as np

import gym
from baselines.common.schedules import LinearSchedule
from baselines.deepq.utils import BatchInput, load_state, save_state
from models.lstm_burn_in.build_graph import build_train
from models.lstm_burn_in.replay_buffer import PrioritizedReplayBuffer, PositiveReplayBuffer

import marlo
from threading import Thread, Lock
from models.lstm_burn_in.utils import reformat_reward, reformat_obs, Worker
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
from baselines import logger as logger_b
from pathlib import Path
from gym.wrappers import Monitor

def act_for_mobchase(act_func, env_name):
    if 'Mobchase' in env_name:
        print('make act func for Mobchase')
        def _act(ob, _lstm_state, stochastic=True, update_eps=-1):
            a = act_func(ob, _lstm_state, stochastic=stochastic, update_eps=update_eps)
            modified_action = [7] if a[0][0] == 5 else a[0]
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
    for i, param in enumerate(env_params):
        env_name = param[0]
        client_pool = param[1]
        join_tokens = marlo.make(env_name,
                                 params={
                                     "client_pool": client_pool,
                                     "agent_names": [
                                         "MarLo-Agent-{}".format(i*2),
                                         "MarLo-Agent-{}".format(i*2+1)
                                     ]
                                 })
        total_join_tokens += join_tokens
    # print("join tokens len: {}".format(len(join_tokens)))
    return total_join_tokens


def eval(
          env_params,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          gamma=1.0,
          param_noise=False,
          max_length=80,
          burn_in_steps=40,
          lstm_units=128,
          saved_model=None,
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
            visible_device_list="2",
            # allocator_type='BFC',
            allow_growth=True
        )
    )
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

        env_name = env_params[0][0]
        num_action = 6 if 'Mobchase' in env_name else 8  # TODO

        worker_act_fs, update_worker_fs, train, update_target, debug = build_train(
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

        # Create the schedule for exploration starting from 1.
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

        # Initialize the parameters and copy them to the target network.
        coord = tf.train.Coordinator()
        if saved_model is None:
            print("Please set the path to model file")
            return
        else:
            print("Load saved model")
            load_state(saved_model)
            act_var = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deepq/q_func"))
            act_var = np.array(act_var)
            eval_save_path = str(Path(saved_model).parent)
            logger_b.configure(dir=eval_save_path)
            np.save(os.path.join(eval_save_path, 'policy.npy'), act_var)
        update_target()

        # marlo set up ----------------------------------------------------
        join_tokens = get_join_tokens(env_params)
        envs = [marlo.init(token) for token in join_tokens]
        envs = [Monitor(env, directory=os.path.join(eval_save_path, 'agent{}'.format(i))) for i, env in enumerate(envs)]
        if not join_tokens:
            return
        _idxes = np.arange(len(join_tokens))
        lock = Lock()

        # training -------------------------------------------------------
        episode_step_queue = Queue(maxsize=1)
        episode_step_queue.put([0, 0])
        return_queue = Queue()
        rb = PrioritizedReplayBuffer(size=64, alpha=0.9, output_length=max_length,
                                      burn_in_length=burn_in_steps)

        # set_worker
        workers = []
        for i in range(len(env_params)):
            _envs = [envs[i*2], envs[i*2+1]]
            env_name = env_params[i][0]
            _join_tokens = [join_tokens[i*2], join_tokens[i*2+1]]
            _agent_ids = [_idxes[i*2], _idxes[i*2+1]]
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

        for worker in workers:
            worker.stop()

        return
