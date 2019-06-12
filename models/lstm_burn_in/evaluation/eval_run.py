import marlo
import os
import json
import numpy as np
from models.lstm_burn_in.evaluation.build_eval_action import build_eval_act
from models.lstm_burn_in.utils import reformat_obs, env_detector
import tensorflow as tf
from gym.wrappers import Monitor
from models.lstm_burn_in.evaluation.build_eval_action import act_reshape
import baselines.common.tf_util as U

def get_join_tokens():
    if marlo.is_grading():
        """
            In the crowdAI Evaluation environment obtain the join_tokens 
            from the evaluator

            the `params` parameter passed to the `evaluator_join_token` only allows
            the following keys : 
                    "seed",
                    "tick_length",
                    "max_retries",
                    "retry_sleep",
                    "step_sleep",
                    "skip_steps",
                    "videoResolution",
                    "continuous_to_discrete",
                    "allowContinuousMovement",
                    "allowDiscreteMovement",
                    "allowAbsoluteMovement",
                    "add_noop_command",
                    "comp_all_commands"
                    # TODO: Add this to the official documentation ? 
                    # Help Wanted :D Pull Requests welcome :D 
        """
        join_tokens = marlo.evaluator_join_token(params={})

    else:
        """
            When debugging locally,
            Please ensure that you have a Minecraft client running on port 10000 and 10001
            by doing : 
            $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000
            $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10001
        """
        ip = '127.0.0.1'
        # ip = '192.168.2.78'
        print("Generating join tokens locally...")
        client_pool = [(ip, 11000), (ip, 11001)]
        join_tokens = marlo.make('MarLo-MobchaseTrain2-v0',
                                 params={
                                     "client_pool": client_pool,
                                     "agent_names": [
                                         "MarLo-Agent-0",
                                         "MarLo-Agent-1"
                                     ]
                                 })

    return join_tokens


def get_join_tokens_my(env_params):
    print("Generating join tokens locally...")
    total_join_tokens = []
    for i, param in enumerate(env_params):
        env_name = param[0]
        client_pool = param[1]
        join_tokens = marlo.make(env_name,
                                 params={
                                     "client_pool": client_pool,
                                     "agent_names": [
                                         "MarLo-Agent-{}".format(i * 2),
                                         "MarLo-Agent-{}".format(i * 2 + 1)
                                     ]
                                 })
        total_join_tokens += join_tokens
    # print("join tokens len: {}".format(len(join_tokens)))
    return total_join_tokens


def get_join_tokens_my_single(env_params):
    print("Generating join tokens locally...")
    total_join_tokens = []
    for i, param in enumerate(env_params):
        env_name = param[0]
        client_pool = param[1]
        join_tokens = marlo.make(env_name,
                                 params={
                                     "client_pool": client_pool,
                                 })
        total_join_tokens += join_tokens
    # print("join tokens len: {}".format(len(join_tokens)))
    return total_join_tokens


@marlo.threaded
# def run_agent(join_token, agent_id, sess, act):
def run_agent(env, agent_id, sess, act):
    """
        Where agent_id is an integral number starting from 0
        In case, you have requested GPUs, then the agent_id will match
        the GPU device id assigneed to this agent.
    """
    # env = marlo.init(join_token)
    # env_detector(env)
    env = Monitor(env, directory='/tmp/agent{}'.format(agent_id), force=True)  # TODO

    def zero_state(lstm_units=512):
        state = np.zeros((2, 1, lstm_units))
        return state

    obs = env.reset()
    obs = reformat_obs(obs)
    done = False
    count = 0
    lstm_state = zero_state()
    with sess.as_default():
        while not done:
            a, lstm_state = act(obs[None], lstm_state)
            obs, reward, done, info = env.step(a[0])
            obs = reformat_obs(obs)
            count += 1
            print("agent_id:{}, action:{}, reward:{}, done:{}, count:{}".format(agent_id, a, reward, done, count))

    # It is important to do this env.close()
    # print(obs.shape, env.action_space)
    env.close()


def run_episode():
    """
    Single episode run
    """
    # get join tokens ------------------------------------
    # join_tokens = get_join_tokens()
    ip = '127.0.0.1'
    # ip = '192.168.2.78'
    base_name = 'MarLo-MobchaseTrain'
    num = 1
    base_port = 11000
    # env_names = [base_name + str(i+1)+'-v0' for i in range(num)]
    # ports = [[(ip, base_port + 10*i), (ip, base_port + 10*i + 1)] for i in range(num)]
    # env_names = ['MarLo-Obstacles-v0']
    env_names = ['MarLo-FindTheGoal-v0']
    ports = [[(ip, base_port)]]

    env_params = list(zip(env_names, ports))

    join_tokens = get_join_tokens_my_single(env_params)
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)

    # When the required number of episodes are evaluated
    # The evaluator returns False for join_tokens
    if not join_tokens:
        return

    # set environments and policies  ------------------------------------
    thread_handlers = []
    """
    NOTE: If instead of a dynamic loop, you hard code the run_agent 
    function calls, then the evaluation of your code will fail in case 
    of a tournament, where multiple submissions can control different agents 
    in the same game. 
    """
    envs = []
    act_fs = []
    agent_ids = []
    params_s = []
    set_params_fs = []
    # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters', "Mobchase2.npy")
    # model_path = '/home/isi/karino/tmp/sentanAI/Apartment/policy.npy'
    model_path = '/home/isi/karino/tmp/sentanAI/FindTheGoal/policy.npy'

    # set environments and networks
    for _idx, token in enumerate(join_tokens):
        env = marlo.init(token)
        env_name = env_detector(env)
        envs.append(env)
        agent_ids.append(_idx)

        if 'Mobchase' in env_name:
            num_action = 6
        elif 'Apartment' in env_name:
            num_action = 6
        else:
            num_action = len(env.action_names[0])
        act_f, set_params_f = build_eval_act(num_actions=num_action, scope=env_name+str(_idx))
        act_f = act_reshape(act_f, env_name)
        act_fs.append(act_f)
        set_params_fs.append(set_params_f)

        # TODO choose proper model file
        # params_s.append(np.load(env_name+'.npy'))
        params_s.append(np.load(model_path))

    # set trained parameters
    with sess.as_default():
        U.initialize()
        for i in range(len(set_params_fs)):
            print("load params {}".format(i))
            set_params_fs[i](*params_s[i])

    # run agents
    for env, _idx, act in zip(envs, agent_ids, act_fs):
        # Run agent-N on a separate thread
        thread_handler = run_agent(env, _idx, sess, act)
        # Accumulate thread handlers
        thread_handlers.append(thread_handler)

    # Wait for  threads to complete or raise an exception
    marlo.utils.join_all(thread_handlers)
    print("Episode Run Complete")


if __name__ == "__main__":
    """
        In case of debugging locally, run the episode just once
        and in case of when the agent is being evaluated, continue 
        running episodes for as long as the evaluator keeps supplying
        join_tokens.
    """
    if not marlo.is_grading():
        print("Running single episode...")
        run_episode()
    else:
        while True:
            run_episode()
