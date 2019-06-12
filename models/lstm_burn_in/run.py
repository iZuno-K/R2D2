from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from models.lstm_burn_in.learn import learn
from models.lstm_burn_in.cnn_fc_lstm_fc import cnn_to_fc_to_lstm_to_fc
from models.lstm_burn_in.cnn_lstm_fc import cnn_to_lstm_to_fc


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure(dir='/tmp/lstm_breakoutR2D2FullRewShape')
    set_global_seeds(args.seed)
    # env = make_atari(args.env)
    # env = bench.Monitor(env, logger.get_dir())
    # env = deepq.wrap_atari_dqn(env)
    agent_num = 16
    envs = [make_atari(args.env) for i in range(agent_num)]
    envs = [deepq.wrap_atari_dqn(e) for e in envs]

    lstm_units = 512
    # model = cnn_to_fc_to_lstm_to_fc(
    model = cnn_to_lstm_to_fc(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        dueling=bool(args.dueling),
    )
    max_length = 80
    act = learn(
        envs,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=int(4 * 1e6 / max_length),
        exploration_initial=1.0,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        learning_starts=10000,
        gamma=0.997,
        prioritized_replay=True,
        train_iter=5,
        max_length=max_length,
        burn_in_steps=40,
        positive_batch_ratio=0.5,
        batch_size=64,
        lstm_units=lstm_units,
        reward_reformat=True,
        saved_model=None,
    )
    # act.save("pong_model.pkl") XXX
    [env.close() for env in envs]


if __name__ == '__main__':
    main()
