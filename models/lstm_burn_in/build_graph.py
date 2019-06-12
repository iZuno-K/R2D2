import tensorflow as tf
import models.lstm_burn_in.baselines_utils as U
from models.lstm_burn_in.utils import reformat_reward_tf, inverse_reformat_reward_tf

def build_act(make_obs_ph, q_func, lstm_units, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
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
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        lstm_input = tf.placeholder(tf.float32, [2, None, lstm_units])
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values, lstm_state = q_func(observations_ph.get(), lstm_input, num_actions, scope="q_func")
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        _act = U.function(
                         inputs=[observations_ph, lstm_input, stochastic_ph, update_eps_ph],
                         outputs=[output_actions, lstm_state],
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        def act(ob, _lstm_state, stochastic=True, update_eps=-1):
            return _act(ob, _lstm_state, stochastic, update_eps)
        return act


def build_train(worker_num, make_obs_ph, q_func, num_actions, optimizer, lstm_units, grad_norm_clipping=None, gamma=1.0,
                double_q=True, scope="deepq", reuse=None, param_noise=False, param_noise_filter_func=None,
                max_length=80, burn_in_steps=40, reward_reformat=False):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
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
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    with tf.device("/device:CPU:0"):
        act_f = build_act(make_obs_ph, q_func, lstm_units, num_actions, scope=scope, reuse=reuse)
        act_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/q_func")
        worker_act_fs = []
        update_worker_funcs = []
        for i in range(worker_num):
            worker_act_f = build_act(make_obs_ph, q_func, lstm_units, num_actions, scope='worker{}'.format(i), reuse=False)
            worker_act_fs.append(worker_act_f)
            worker_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="worker{}".format(i)+"/q_func")
            update_worker_net = []
            for a_var, w_var in zip(sorted(act_var, key=lambda v: v.name),
                                       sorted(worker_var, key=lambda v: v.name)):
                update_worker_net.append(w_var.assign(a_var))
            update_worker_net = tf.group(*update_worker_net)
            update_worker_funcs.append(U.function([], [], updates=[update_worker_net]))

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        # TODO None=batch*max_length
        obs_input = make_obs_ph("obs")  # (batch*max_length, dim)
        obs_burn_in = make_obs_ph("obs")  # (batch*burn_in_steps, dim)
        lstm_input = tf.placeholder(tf.float32, [2, None, lstm_units])  # None=batch
        act_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_p1_input = make_obs_ph("obsp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")
        seq_length_ph = tf.placeholder(tf.float32, [None], name="seq_length")
        seq_length_burn_in_ph = tf.placeholder(tf.float32, [None], name="seq_length_burn_in")

        # burn in network
        _, lstm_states = q_func(obs_burn_in.get(), lstm_input, num_actions, scope="q_func", reuse=True, max_length=burn_in_steps, seq_length_inpt=seq_length_burn_in_ph)

        lstm_states = tf.stop_gradient(lstm_states)

        # q network evaluation
        q_t, _ = q_func(obs_input.get(), lstm_states, num_actions, scope="q_func", reuse=True, max_length=max_length, seq_length_inpt=seq_length_ph)  # reuse parameters from act
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1, _ = q_func(obs_p1_input.get(), lstm_states, num_actions, scope="target_q_func", max_length=max_length, seq_length_inpt=seq_length_ph)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net, _ = q_func(obs_input.get(), lstm_input, num_actions, scope="q_func", reuse=True, max_length=max_length, seq_length_inpt=seq_length_ph)  # reuse parameters from act
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        if reward_reformat:
            q_t_selected_target = reformat_reward_tf(rew_ph + gamma * inverse_reformat_reward_tf(q_tp1_best_masked))
        else:
            q_t_selected_target = rew_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        mask = tf.sequence_mask(seq_length_ph, max_length, dtype=tf.float32)  # (batch, max_length)
        mask = tf.reshape(mask, (-1,))
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)  # (batch*max_length,)
        td_error = td_error * mask  # mask padding
        errors = U.huber_loss(td_error)
        td_error = tf.reshape(td_error, (-1, max_length))
        # weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        weighted_error = tf.reduce_sum(errors) / tf.reduce_sum(seq_length_ph)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_input,
                lstm_input,
                act_ph,
                rew_ph,
                obs_p1_input,
                done_mask_ph,
                # importance_weights_ph
                seq_length_ph,
                obs_burn_in,
                seq_length_burn_in_ph,
            ],
            outputs=[td_error, weighted_error],
            updates=[optimize_expr]
        )

        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_input], q_t * mask)

        return act_f, worker_act_fs, update_worker_funcs, train, update_target, {'q_values': q_values}
