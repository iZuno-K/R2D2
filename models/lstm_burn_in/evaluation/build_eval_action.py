from models.lstm_burn_in.build_graph import build_act
from models.lstm_burn_in.utils import reformat_obs
from lxml import etree
import tensorflow as tf
import numpy as np
from models.lstm_burn_in.cnn_lstm_fc import cnn_to_lstm_to_fc
from baselines.deepq.utils import BatchInput
import baselines.common.tf_util as U

def act_reshape(act_func, env_name):
    if 'Mobchase' in env_name:
        print('make act func for Mobchase')
        def _act(ob, _lstm_state, stochastic=False, update_eps=-1):
            a = act_func(ob, _lstm_state, stochastic=stochastic, update_eps=update_eps)
            modified_action = [7] if a[0][0] == 5 else a[0]
            a[0] = modified_action
            return a  # [action, lstm]
    if 'Apartment' in env_name:
        print('make act for obstacle')
        def _act(ob, _lstm_state, stochastic=False, update_eps=-1):
            a = act_func(ob, _lstm_state, stochastic=stochastic, update_eps=update_eps)
            modified_action = [a[0][0] + 1]
            a[0] = modified_action
            return a  # [action, lstm]

    else:
        def _act(ob, _lstm_state, stochastic=False, update_eps=-1):
            a = act_func(ob, _lstm_state, stochastic=stochastic, update_eps=update_eps)
            return a  # [action, lstm, ]

    return _act


def build_eval_act(num_actions, scope, lstm_units=512):
    observation_space_shape = (90, 120, 3)
    def make_obs_ph(name):
        return BatchInput(observation_space_shape, name=name)

    with tf.device("/device:CPU:0"):
        q_func = cnn_to_lstm_to_fc(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[512],
            dueling=True,
        )
        act_f  = build_act(make_obs_ph=make_obs_ph, q_func=q_func, lstm_units=lstm_units, num_actions=num_actions, scope=scope, reuse=False)
        act_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/q_func")
        phs = []
        updates_tf = []
        for act_var in act_vars:
            ph = tf.placeholder(tf.float32, act_var.shape)
            phs.append(ph)
            update = act_var.assign(ph)
            updates_tf.append(update)

    updates_tf = tf.group(*updates_tf)
    set_params_f = U.function(
        inputs=phs,
        outputs=[],
        updates=[updates_tf]
    )

    return act_f, set_params_f

if __name__ == '__main__':
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
    act_f = build_eval_act(sess=sess, model_path='/home/isi/karino/tmp/marlo/r2d2/3/policy.npy', num_actions=6, scope='mobchase')
    print('1 is done')
    act_f2 = build_eval_act(sess=sess, model_path='/home/isi/karino/tmp/marlo/r2d2/3/policy.npy', num_actions=6, scope='mobchase2')
    print('2 is done')


