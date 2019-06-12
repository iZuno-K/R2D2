import tensorflow as tf
import numpy as np

def lstm_state_debug():
    # (2, batch, seq_length)
    length = 5
    batch_size = 2
    ph1 = tf.placeholder(tf.float32, [None, length, 2], name='ph1')
    ph_length = tf.placeholder(tf.int32, [None,], name='ph_length')
    lstm = tf.nn.rnn_cell.LSTMCell(num_units=8)
    lstm_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
    lstm_out, lstm_state = tf.nn.dynamic_rnn(cell=lstm, inputs=ph1,
                                             sequence_length=ph_length,
                                             initial_state=lstm_state)  # (batch, max_words, )
    lengths = np.arange(batch_size) + 1
    inpt = np.arange(batch_size*length*2, dtype=np.float32).reshape(batch_size, length, 2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed = {ph1: inpt, ph_length: lengths}
        out1, out2 = sess.run(lstm_state, feed)
        pass

    # only the final out put is available

if __name__ == '__main__':
    lstm_state_debug()