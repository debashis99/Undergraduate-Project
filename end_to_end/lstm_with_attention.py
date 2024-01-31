import tensorflow as tf
import collections
from tensorflow.contrib.rnn import RNNCell, LSTMCell, LSTMStateTuple


LstmAttentionState = collections.namedtuple("LstmAttentionState", ("lstm_state", "f_hd_atn"))

class LstmAttentionCell(RNNCell):
    def __init__(self,num_units, img , num_proj, dtype=tf.float32):
        """
        Args:
            cell: (RNNCell)
            attention_mechanism: (AttentionMechanism)
            dropout: (tf.float)
            attn_cell_config: (dict) hyper params
        """
        self._cell = LSTMCell(num_units, state_is_tuple=True)
        self._img = img
        self.init_h_state = self.initial_h_state("f_hd_atn", 512)
    
        self._dim_e          = 512
        self._dim_o          = 512
        self._num_units      = num_units
        self._num_proj       = num_proj
        self._dtype          = dtype

        # for RNNCell
        self._state_size = LstmAttentionState(self._cell._state_size, self._dim_o)


    @property
    def state_size(self):
        return self._state_size


    @property
    def output_size(self):
        return self._num_proj


    @property
    def output_dtype(self):
        return self._dtype


    def initial_cell_state(self, cell):
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self.initial_h_state(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell


    def initial_h_state(self, name, dim):
        """Returns initial state of dimension specified by dim"""

        init = lambda shape: np.random.rand(*shape)
    
        img_mean = tf.reduce_mean(self._img, axis=1)
        W = tf.get_variable("W_{}_0".format(name), shape=[512, dim],dtype=tf.float32)
        b = tf.get_variable("b_{}_0".format(name), shape=[dim], dtype=tf.float32)
        h = tf.tanh(tf.matmul(img_mean, W) + b)

        return h

    def initial_state(self):
        """Returns initial state for the lstm"""

        initial_cell_state = self.initial_cell_state(self._cell)
        initial_o          = self.initial_h_state("f_hd_atn", 512)

        return LstmAttentionState(initial_cell_state, initial_o)
    

    def __call__(self, inp, state):
        prev_lstm_state, prev_f_hd_atn = state
        x = tf.concat([inp, prev_f_hd_atn], axis=-1)

        new_h, new_lstm_state = self._cell.__call__(x, prev_lstm_state)

        ### computing attention as f(init_h_state, new_h)

        # over the image, shape = (batch size, n, 512)
        W1_e = tf.layers.dense(inputs=self.init_h_state, units=512, use_bias=False)
        # over the hidden vector, shape = (batch size, 512)
        W2_h = tf.layers.dense(inputs=new_h, units=512, use_bias=False)

        # sums the two contributions
        n = tf.shape(self.init_h_state)[1]
        a = tf.tanh(W1_e + tf.expand_dims(W2_h, axis=1))
        beta = tf.get_variable("beta", shape=[512, 1], dtype=tf.float32)
        a_flat = tf.reshape(a, shape=[-1, 512])
        a_flat = tf.matmul(a_flat, beta)
        a = tf.reshape(a, shape=[-1, n])

        # compute weights
        a = tf.nn.softmax(a)
        a = tf.expand_dims(a, axis=-1)
        c = tf.reduce_sum(a * self.init_h_state, axis=1)


        # compute f_hd_atn (use for predict token)
        W3_o = tf.layers.dense(inputs=tf.concat([new_h, c], axis=-1), units=512, use_bias=False)
        new_f_hd_atn = tf.tanh(W3_o)

        # compute the logits scores (before softmax)
        logits = tf.layers.dense(inputs=new_lstm_state, units=self._num_proj, use_bias=False)

        new_state = LstmAttentionState(new_lstm_state, new_f_hd_atn)
        return (logits, new_state)



