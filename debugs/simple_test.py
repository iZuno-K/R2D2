import marlo
import gym.spaces
import gym.wrappers as wrappers
from lxml import etree

# ip = '127.0.0.1'
ip = '192.168.2.78'
client_pool = [(ip, 11000)]
# env_name = 'MarLo-Obstacles-v0'
# env_name = 'MarLo-Vertical-v0'
# env_name = 'MarLo-FindTheGoal-v0'
# env_name = 'MarLo-CliffWalking-v0'
env_name = 'MarLo-FindTheGoal-v0'
join_tokens = marlo.make(env_name,
                          params={
                            "client_pool": client_pool
                          })
# As this is a single agent scenario,
# there will just be a single token
assert len(join_tokens) == 1
join_token = join_tokens[0]

env = marlo.init(join_token)
# env = wrappers.Monitor(env, "/tmp/marlo")
print(env.action_names)
print(len(env.action_names))
print(len(env.action_names[0]))
xml = etree.fromstring(env.params["mission_xml"].encode())
summary = xml.find('{http://ProjectMalmo.microsoft.com}About/'
                   '{http://ProjectMalmo.microsoft.com}Summary').text
print(summary)
done = False
print("start step")
# for i in range(10):
observation = env.reset()
while not done:
    _action = env.action_space.sample()
    # obs, reward, done, info = env.step(_action)
    obs, reward, done, info = env.step(0)
    print("reward:", reward)
    #print("done:", done)
    #print("info", info)
env.close()
