import os
import sys
package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
package_path = os.path.abspath(package_path)
print(package_path)
sys.path.append(package_path)
from models.lstm_burn_in.baselines_utils import set_global_seeds
import argparse
from models.lstm_burn_in import baselines_logger as logger_b
from models.lstm_burn_in.learn_marlo import learn
from models.lstm_burn_in.cnn_lstm_fc import cnn_to_lstm_to_fc

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger_b.configure(dir='/tmp/FindTheGoal_noPosRB')
    # logger_b.configure(dir='/home/isi/karino/tmp/marlo/marlo_on_gaia1')

    set_global_seeds(args.seed)

    # environment setting
    ip = '127.0.0.1'
    # ip = '192.168.2.78'
    # base_name = 'MarLo-MobchaseTrain'
    # base_name = 'MarLo-TreasurehuntTrain'
    # num = 5
    # base_port = 11000
    # env_names = [base_name + str(i+1)+'-v0' for i in range(num)]
    # ports = [[(ip, base_port + 10*i), (ip, base_port + 10*i + 1)] for i in range(num)]

    # base_name = 'MarLo-Obstacles-v0'
    # base_name = 'MarLo-CliffWalking-v0'
    base_name = 'MarLo-FindTheGoal-v0'
    # env_names = ['MarLo-Obstacles-v0', 'MarLo-FindTheGoal-v0']
    num = 5
    base_port = 11000
    env_names = [base_name for i in range(num)]
    ports = [[(ip, base_port + i)] for i in range(num)]

    env_params = list(zip(env_names, ports))
    print("environment parameters")
    print(env_params)

    # model setting
    lstm_units = 512
    # model = cnn_to_fc_to_lstm_to_fc(
    model = cnn_to_lstm_to_fc(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        dueling=True,
    )
    max_length = 80
    act = learn(
        q_func=model,
        lr=1e-4,
        # max_timesteps=args.num_timesteps,
        max_timesteps=int(50e6),
        # buffer_size=int(4 * 1e6 / max_length),
        buffer_size=int(4 * 1e5 / max_length),
        exploration_initial=1.0,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        # train_freq=4,
        learning_starts=10000,  # (should be > max_length * batch_size)
        # target_network_update_freq=1000,
        gamma=0.997,
        prioritized_replay=True,
        max_length=max_length,
        burn_in_steps=40,
        train_iter=5,
        initial_positive_batch_ratio=0.8,
        batch_size=64,
        lstm_units=lstm_units,
        reward_reformat=True,
        ip=ip,
        saved_model=None,
        env_params=env_params,
    )
    # act.save("pong_model.pkl") XXX
    # env.close()


if __name__ == '__main__':
    main()
