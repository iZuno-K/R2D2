import tensorflow as tf
import marlo


def get_join_tokens():
    print("Generating join tokens locally...")
    client_pool = [('127.0.0.1', 11000), ('127.0.0.1', 11001)]
    join_tokens = marlo.make('MarLo-MobchaseTrain1-v0',
                             params={
                                 "client_pool": client_pool,
                                 "agent_names": [
                                     "MarLo-Agent-0",
                                     "MarLo-Agent-1"
                                 ]
                             })
    return join_tokens

@marlo.threaded
def run_agent(join_token, agent_id):
    env = marlo.init(join_token)
    observation = env.reset()
    done = False
    count = 0
    while not done:
        _action = env.action_space.sample()
        obs, reward, done, info = env.step(_action)
        print("agent_id:{}, reward:{}, done:{}, info:{}".format(agent_id, reward, done, info))

    # It is important to do this env.close()
    print(observation.shape, env.action_space)
    env.close()


def main():
    join_tokens = get_join_tokens()

    # Main thread: create a coordinator.
    coord = tf.train.Coordinator()

    thread_handlers = []
    for _idx, join_token in enumerate(join_tokens):
        # Run agent-N on a separate thread
        thread_handler = run_agent(join_token, _idx)

        # Accumulate thread handlers
        thread_handlers.append(thread_handler)
