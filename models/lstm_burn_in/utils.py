import tensorflow as tf  # pylint: ignore-module
import collections
import threading
import time
import numpy as np
import marlo

from models.lstm_burn_in import baselines_logger as logger
import os
import csv
from lxml import etree
from PIL import Image

ALREADY_INITIALIZED = set()

def initialize(sess):
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    sess.run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

# ================================================================
# Mathematical utils
# ================================================================

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


# ================================================================
# Theano-like Function
# ================================================================

def function(sess, inputs, outputs, updates=None, givens=None):
    """Just like Theano function. Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    Parameters
    ----------
    inputs: [tf.placeholder, tf.constant, or object with make_feed_dict method]
        list of input arguments
    outputs: [tf.Variable] or tf.Variable
        list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    """
    if isinstance(outputs, list):
        return _Function(sess, inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(sess, inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(sess, inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, sess, inputs, outputs, updates, givens):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.sess = sess
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = self.sess.run(self.outputs_update, feed_dict=feed_dict)[:-1]
        return results


class BatchMakeThread(object):
    def __init__(self, queue, replay_buffer, batch_size, lock):
        """
        :param queue:
        :param replay_buffer:
        :param batch_size:
        :param lock:
        :return:
        """
        self.queue = queue
        self.rb = replay_buffer
        self.batch_size = batch_size
        self.lock = lock
        self.stop_event = threading.Event()
        self.start_event = threading.Event()
        self._thread = threading.Thread(target=self._batch_make_thread)

        self._thread.start()

    def _batch_make_thread(self):
        while not self.stop_event.is_set():
            if self.start_event.is_set() and len(self.rb) > self.batch_size and not self.queue.full():
                    with self.lock:
                        # start = time.time()
                        # print("put queue batch... ")
                        self.queue.put(self.rb.sample(self.batch_size, beta=0.4))
                        # print("put queue batch done ({}s)".format(time.time() - start))

    def start(self):
        self.start_event.set()

    def stop(self):
        self.stop_event.set()
        self._thread.join()

def reformat_obs(observation):
    observation = Image.fromarray(observation)
    observation = observation.resize((120, 90), Image.BILINEAR)
    observation = np.array(observation).astype(np.float32) / 255.0
    observation = LazyFrame(observation)
    return observation

_eps = 1e-2
def reformat_reward(reward):
    x = reward
    if x == 0:
        x = -0.02
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + _eps * x

def reformat_reward_tf(rew_tf):
    x = rew_tf
    return tf.sign(x) * (tf.sqrt(tf.abs(x) + 1) - 1) + _eps * x

def inverse_reformat_reward_tf(q_tf):
    y = q_tf
    return tf.sign(q_tf) * ((tf.square((tf.sqrt(1+4*_eps*(tf.abs(y)+1+_eps)) - 1) / 2. / _eps )) - 1)


class LazyFrame(object):
    def __init__(self, frame):
        """modified version from baselines implementation"""
        self._frame = frame
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.array(self._frame)
            self._frame = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class Worker(object):
    def __init__(self, envs, env_name, join_tokens, agent_ids, sess, coord, act, update, zero_state, reward_reformat,
                 exploration_scheduler, buffer, lock, episode_and_step_queue, return_queue):
        # two agents
        self.envs = envs
        self.env_name = env_name
        self.join_tokens = join_tokens
        self.agent_ids = agent_ids

        self.sess = sess
        self.coord = coord
        self.act = act  # function
        self.update = update # function
        self.zero_state = zero_state  # function
        self.reward_reformat = reward_reformat
        self.exploration = exploration_scheduler
        self.t = 0

        self.buffer = buffer
        self.lock = lock
        self.ep_st_queue = episode_and_step_queue  # maxsize = 1
        self.return_queue = return_queue

        self.stop_event = threading.Event()

        td = logger.get_dir()
        os.makedirs(td, exist_ok=True)
        self.f = open(os.path.join(td, "log_{}{}.csv".format(env_name, agent_ids[0])), 'w')
        self.writer = csv.writer(self.f, lineterminator='\n', delimiter=',')
        self.writer.writerow(['episode, ret, steps, epsilon'])
        self.f.flush()

        self._thread = threading.Thread(target=self.worker_thread)

        self._thread.start()

    @marlo.threaded
    def run_agent(self, env, agent_id, kwargs):
        """
            Where agent_id is an integral number starting from 0
            In case, you have requested GPUs, then the agent_id will match
            the GPU device id assigneed to this agent.
        """
        with self.sess.as_default():
            # print("{}: {}".format(agent_id, tf.get_default_session()))

            # logger.warn("thread in")
            obs = env.reset()
            # logger.warn("start episode")
            obs = reformat_obs(obs)

            done = False
            ep_obs = []
            ep_lstm = []
            lstm_state = self.zero_state(1)
            ep_action = []
            ep_rew = []
            ep_obs1 = []
            ep_done = []
            while not done:
                ep_obs.append(obs)
                ep_lstm.append(np.squeeze(lstm_state, axis=1))
                # logger.warn("squeeze done")
                # logger.warn("obs shape {}".format(np.array(obs).shape))

                # action = env.action_space.sample()
                _action, lstm_state, modified_action = self.act(np.array(obs)[None], lstm_state, **kwargs)
                # logger.warn("action done")
                # logger.warn("type action{}, modified{}".format(type(_action), type(modified_action)))
                # logger.warn("len action{}, modified{}".format(len(_action), len(modified_action)))

                action = _action[0]
                next_obs, reward, done, info = env.step(modified_action[0])
                # logger.warn("step done")

                next_obs = reformat_obs(next_obs)
                reward = reformat_reward(reward) if self.reward_reformat else reward
                # logger.warn("reformat done")

                ep_action.append(action)
                ep_rew.append(reward)
                ep_obs1.append(next_obs)
                ep_done.append(float(done))
                obs = next_obs

                # logger.warn("reward(agent{}): {}, action{}".format(agent_id, reward, action))
            with self.lock:
                # logger.warn("RB appending ...")
                self.buffer.add_path(ep_obs, ep_lstm, ep_action, ep_rew, ep_obs1, ep_done)
                # logger.warn("RB append done")
            return ep_rew


    def worker_thread(self):
        episode = 0
        total_t = 0
        param_update_freq = 2500
        _param_update_freq = 2500
        while not self.stop_event.is_set():
            thread_handlers = []
            queues = []
            t = 0
            update_eps = self.exploration.value(total_t)
            kwargs = {'stochastic': True, 'update_eps':update_eps}

            for env, id in zip(self.envs, self.agent_ids):
                thread_handler, queue = self.run_agent(env, id, kwargs)
                thread_handlers.append(thread_handler)
                queues.append(queue)
            [th.join() for th in thread_handlers]
            # print("worked time", time.time() - tmp)
            # marlo execution -----------------------------------------

            # extract reward info and logging
            sum_returns = 0.
            sum_lengths = 0
            valid_ep = 0
            for q in queues:
                ep_rew = q.get()
                if type(ep_rew) == list:
                    sum_returns += sum(ep_rew)
                    t += len(ep_rew)
                    valid_ep += 1
            if valid_ep > 0:
                averaged_return = sum_returns / valid_ep
                [_ep, _t] = self.ep_st_queue.get()
                _ep += 1
                _t += t
                total_t = _t
                episode = _ep
                self.ep_st_queue.put([_ep, _t])
                self.writer.writerow([_ep, averaged_return, _t, update_eps])
                self.return_queue.put(averaged_return)
                self.f.flush()

                if _t > _param_update_freq:
                    with self.sess.as_default():
                        self.update()
                    _param_update_freq = _t + param_update_freq

    def stop(self):
        self.stop_event.set()
        self._thread.join()
        self.f.close()


def env_detector(env):
    xml = etree.fromstring(env.params["mission_xml"].encode())
    summary = xml.find('{http://ProjectMalmo.microsoft.com}About/'
                       '{http://ProjectMalmo.microsoft.com}Summary').text
    env_name = None
    if "Build Battle" in summary:
        print("Let's build battle!")
        env_name = "Buildbattle"
    elif "Treasure Hunt" in summary:
        print("Treasure hunting we go!")
        env_name = "Treasurehunt"
    elif "Catch" in summary:
        print("Help catch the mob!")
        env_name = "Mobchase"
    elif "apartment" in summary:
        print("Find the goal! The apartment!")
        env_name = "Apartment"
    elif "Cliff" in summary:
        print("Cliff walking mission based on Sutton and Barto.")
        env_name = "Cliff"
    else:
        print(summary)
        env_name = 'Other'

    return env_name