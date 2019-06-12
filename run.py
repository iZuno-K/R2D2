import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import marlo
import json
import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from models.lstm_burn_in.evaluation.build_eval_action import build_eval_act, act_reshape
from models.lstm_burn_in.utils import reformat_obs, env_detector


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
        # env = 'MarLo-MobchaseTrain2-v0'
        # env = "MarLo-BuildbattleTrainX-v0"
        # env=  "MarLo-TreasurehuntTrainX-v0"
        join_tokens = marlo.make('MarLo-TreasurehuntTrain1-v0',
                                 params={
                                    "client_pool": client_pool,
                                    "agent_names" : [
                                        "MarLo-Agent-0",
                                        "MarLo-Agent-1"
                                    ]
                                 })
    return join_tokens


@marlo.threaded
def run_agent(env, agent_id, sess, act):
    """
        Where agent_id is an integral number starting from 0
        In case, you have requested GPUs, then the agent_id will match
        the GPU device id assigneed to this agent.
    """
    def zero_state(lstm_units=512):
        state = np.zeros((2, 1, lstm_units))
        return state

    obs = env.reset()
    obs = reformat_obs(obs)
    done = False
    # count = 0
    lstm_state = zero_state()
    with sess.as_default():
        while not done:
            a, lstm_state = act(obs[None], lstm_state)
            obs, reward, done, info = env.step(a[0])
            obs = reformat_obs(obs)
            # count += 1
            # print("agent_id:{}, action:{}, reward:{}, done:{}, count:{}".format(agent_id, a, reward, done, count))

    # It is important to do this env.close()
    env.close()


def run_episode():
    """
    Single episode run
    """
    join_tokens = get_join_tokens()

    # When the required number of episodes are evaluated
    # The evaluator returns False for join_tokens
    if not join_tokens:
        return

    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)

    """
    NOTE: If instead of a dynamic loop, you hard code the run_agent 
    function calls, then the evaluation of your code will fail in case 
    of a tournament, where multiple submissions can control different agents 
    in the same game. 
    """
    # set environments and policies  ------------------------------------
    thread_handlers = []
    envs = []
    act_fs = []
    agent_ids = []
    params_s = []
    set_params_fs = []
    # set environments and networks
    param_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/lstm_burn_in/evaluation/parameters')
    print(param_dir)
    for _idx, token in enumerate(join_tokens):
        env = marlo.init(token)
        env_name = env_detector(env)  # Buildbattle, Treasurehunt or Mobchase
        envs.append(env)
        agent_ids.append(_idx)

        num_action = 6 if env_name == "Mobchase" else 8
        act_f, set_params_f = build_eval_act(num_actions=num_action, scope=env_name+str(_idx))
        act_f = act_reshape(act_f, env_name)
        act_fs.append(act_f)
        set_params_fs.append(set_params_f)

        param_file = os.path.join(param_dir, env_name+'.npy')
        params_s.append(np.load(param_file))

    # set trained parameters
    with sess.as_default():
        U.initialize()
        for i in range(len(set_params_fs)):
            # print("load params {}".format(i))
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
