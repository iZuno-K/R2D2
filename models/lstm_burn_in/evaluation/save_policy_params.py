import os
import tensorflow as tf
import numpy as np

from baselines.deepq.utils import BatchInput, load_state
from models.lstm_burn_in.build_graph import build_train

from pathlib import Path
from models.lstm_burn_in.cnn_lstm_fc import cnn_to_lstm_to_fc


def save_polic_param(
        env_params,
        q_func,
        lr=5e-4,
        gamma=1.0,
        param_noise=False,
        max_length=80,
        burn_in_steps=40,
        lstm_units=512,
        saved_model=None,
        reward_reformat=True,):
    # Create all the functions necessary to train the model
    config = tf.ConfigProto(device_count = {'GPU': 0}
        # gpu_options=tf.GPUOptions(
        #     device_count={'GPU': 0},
        #     # visible_device_list="-1",
        #     # allocator_type='BFC',
        #     # allow_growth=True
        # )
    )
    sess = tf.Session(config=config)

    with sess.as_default():
        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph
        observation_space_shape = (90, 120, 3)

        def make_obs_ph(name):
            return BatchInput(observation_space_shape, name=name)

        env_name = env_params[0][0]
        if 'Mobchase' in env_name:
            num_action = 6
        elif 'Treasurehunt' in env_name or 'Buildbattle' in env_name:
            num_action = 8
        else:
            num_action = 5

        act_tf, *_ = build_train(
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

        # Initialize the parameters and copy them to the target network.
        if saved_model is None:
            print("Please set the path to model file")
            return
        else:
            print("Load saved model")
            load_state(saved_model)
            act_var = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deepq/q_func"))
            act_var = np.array(act_var)

            eval_save_path = str(Path(saved_model).parent)
            np.save(os.path.join(eval_save_path, 'policy.npy'), act_var)
            # np.save(os.path.join('/home/isi/karino', 'policy.npy'), act_var)
            print("save donne")

        return


if __name__ == '__main__':
    # saved_model = '/home/isi/karino/tmp/marlo/reward_reshape/0/model'
    # saved_model = '/home/isi/y-suzuki/tmp/marlo/mobchase/1231/model'
    # base_name = 'MarLo-MobchaseTrain'
    # saved_model = '/home/isi/karino/tmp/marlo/Treasurehunt/0/model'
    # saved_model = '/home/isi/karino/tmp/marlo/Treasurehunt/PIL/2/model'
    # base_name = 'MarLo-TreasurehuntTrain'
    # saved_model = '/home/isi/y-suzuki/tmp/marlo/reward_reshape/build_battle/model'
    # base_name = 'MarLo-BuildbattleTrain'
    # saved_model = '/home/isi/karino/tmp/sentanAI/Apartment/model'
    # base_name = 'MarLo-Obstacles-v0'
    saved_model = '/home/isi/karino/tmp/sentanAI/FindTheGoal/model'
    base_name = 'MarLo-FindTheGoal-v0'

    ip = '127.0.0.1'
    num = 5
    base_port = 11000
    # env_names = [base_name + str(i + 1) + '-v0' for i in range(num)]
    # ports = [[(ip, base_port + 10 * i), (ip, base_port + 10 * i + 1)] for i in range(num)]
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
    save_polic_param(env_params=env_params, q_func=model, gamma=0.997, max_length=max_length,
                     burn_in_steps=40, lstm_units=lstm_units, saved_model=saved_model, reward_reformat=True)