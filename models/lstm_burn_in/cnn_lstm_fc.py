import tensorflow as tf
import tensorflow.contrib.layers as layers


def _cnn_to_fc_to_lstm_to_fc(convs, hiddens, dueling, inpt, lstm_inpt, num_actions, scope, max_length=None,
                             seq_length_inpt=None, reuse=False, layer_norm=False):
    """"
    :param convs: [(channel, kernel_size, stride), ...]
    :param hiddens:
    :param dueling:
    :param inpt:
    :param lstm_inpt: lstm_state (2, batch, lstm_units)
    :param num_actions:
    :param scope:
    :param reuse:
    :param layer_norm:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        conv_out = inpt  # (batch, state_dim) or (batch*max_length, state_dim)
        with tf.variable_scope("convnet"):
            for channel, kernel_size, stride in convs:
                conv_out = tf.layers.conv2d(conv_out,
                                       filters=channel,
                                       kernel_size=kernel_size,
                                       strides=stride,
                                       activation=tf.nn.relu)
            conv_out = tf.layers.flatten(conv_out)

        with tf.variable_scope("lstm"):
            lstm_units_dim = lstm_inpt.shape[2]
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units_dim)
            lstm_state = tf.nn.rnn_cell.LSTMStateTuple(lstm_inpt[0], lstm_inpt[1])

            if max_length is None:  # one step execution
                # lstm_out, lstm_state = lstm(fc_out, lstm_state)
                lstm_out, lstm_state = tf.nn.dynamic_rnn(cell=lstm, inputs=tf.expand_dims(conv_out, 1), initial_state=lstm_state)
                lstm_out = tf.squeeze(lstm_out, 1)
            else:  # sequential inputs (training phase)
                state_dim = conv_out.shape[1]
                conv_out = tf.reshape(conv_out, (-1, max_length, state_dim))  # (batch, max_length, state_dim)
                lstm_out, lstm_state = tf.nn.dynamic_rnn(cell=lstm, inputs=conv_out,
                                                         sequence_length=seq_length_inpt,
                                                         initial_state=lstm_state)  # (batch, max_words, )
                lstm_out = tf.reshape(lstm_out, (-1, lstm_units_dim))  # (batch*max_length, lstm_units_dim)

        with tf.variable_scope("action_value"):
            action_out = lstm_out
            for hidden in hiddens:
                action_out = tf.layers.dense(action_out, units=hidden, activation=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = tf.layers.dense(action_out, units=num_actions, activation=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = lstm_out
                for hidden in hiddens:
                    state_out = tf.layers.dense(action_out, units=hidden, activation=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = tf.layers.dense(state_out, units=1, activation=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out, lstm_state  # (batch*max_length,), LSTMtuple(2, batch, max_length, lstm_units)


def cnn_to_lstm_to_fc(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_fc_to_lstm_to_fc(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)
