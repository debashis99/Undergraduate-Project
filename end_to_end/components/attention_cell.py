import tensorflow as tf
import collections
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple


AttentionState = collections.namedtuple("AttentionState", ("cell_state", "o"))


class AttentionCell(RNNCell):
    def __init__(self, cell, num_proj,img, tiles=1, dtype=tf.float32):
        # variables and tensors
        self._cell                = cell
        self._dropout             = 1
        self._img = img

        # hyperparameters and shapes
        self._n_channels     = 512
        self._dim_e          = 512
        self._dim_o          = 512
        self._num_units      = 512
        self._dim_embeddings = 80
        self._num_proj       = num_proj
        self._dtype          = dtype

        # for RNNCell
        self._state_size = AttentionState(self._cell._state_size, self._dim_o)


        if len(img.shape) == 3:
            self._img = img
        else:
            print("Image shape not supported")
            raise NotImplementedError

        # dimensions
        self._n_regions  = tf.shape(self._img)[1]
        self._tiles      = tiles
        # attention vector over the image
        self._att_img = tf.layers.dense(
            inputs=self._img,
            units=512,
            use_bias=False,
            name="att_img")



    @property
    def state_size(self):
        return self._state_size


    @property
    def output_size(self):
        return self._num_proj


    @property
    def output_dtype(self):
        return self._dtype


    def initial_state(self):
        """Returns initial state for the lstm"""
        initial_cell_state = self.initial_cell_state(self._cell, self._img)
        initial_o          = self.initial_h_state("o", self._dim_o, self._img)

        return AttentionState(initial_cell_state, initial_o)

    
    def initial_cell_state(self,cell,img):
        """Returns initial state of a cell computed from the image

        Assumes cell.state_type is an instance of named_tuple.
        Ex: LSTMStateTuple

        Args:
            cell: (instance of RNNCell) must define _state_size

        """
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self.initial_h_state(hidden_name, hidden_dim,img)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell


    def initial_h_state(self,name, dim, img):
        """Returns initial state of dimension specified by dim"""
        with tf.variable_scope("attn_cell"):
            img_mean = tf.reduce_mean(img, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[512 ,dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h

    def context(self, h, tiles,regions, _img, _att_img, ):
        with tf.variable_scope("attn_cell"):
            if tiles > 1:
                att_img = tf.expand_dims(_att_img, axis=1)
                att_img = tf.tile(att_img, multiples=[1, tiles, 1, 1])
                att_img = tf.reshape(att_img, shape=[-1, regions, 512])
                img = tf.expand_dims(_img, axis=1)
                img = tf.tile(img, multiples=[1, tiles, 1, 1])
                img = tf.reshape(img, shape=[-1, regions,512])
            else:
                att_img = _att_img
                img     = _img

            # computes attention over the hidden vector
            att_h = tf.layers.dense(inputs=h, units= 512, use_bias=False)

            # sums the two contributions
            att_h = tf.expand_dims(att_h, axis=1)
            att = tf.tanh(att_img + att_h)

            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            att_beta = tf.get_variable("att_beta", shape=[512, 1],
                    dtype=tf.float32)
            att_flat = tf.reshape(att, shape=[-1, 512])
            e = tf.matmul(att_flat, att_beta)
            e = tf.reshape(e, shape=[-1, regions])

            # compute weights
            a = tf.nn.softmax(e)
            a = tf.expand_dims(a, axis=-1)
            c = tf.reduce_sum(a * img, axis=1)

            return c


    def step(self, embedding, attn_cell_state):
        """
        Args:
            embedding: shape = (batch_size, dim_embeddings) embeddings
                from previous time step
            attn_cell_state: (AttentionState) state from previous time step

        """
        prev_cell_state, o = attn_cell_state

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # compute new h
            x                     = tf.concat([embedding, o], axis=-1)
            new_h, new_cell_state = self._cell.__call__(x, prev_cell_state)
            new_h = tf.nn.dropout(new_h, self._dropout)

            # compute attention
            c = self.context(new_h,self._tiles,self._n_regions, self._img, self._att_img)

            # compute o
            o_W_c = tf.get_variable("o_W_c", dtype=tf.float32,
                    shape=(self._n_channels, self._dim_o))
            o_W_h = tf.get_variable("o_W_h", dtype=tf.float32,
                    shape=(self._num_units, self._dim_o))

            new_o = tf.tanh(tf.matmul(new_h, o_W_h) + tf.matmul(c, o_W_c))
            new_o = tf.nn.dropout(new_o, self._dropout)

            y_W_o = tf.get_variable("y_W_o", dtype=tf.float32,
                    shape=(self._dim_o, self._num_proj))
            logits = tf.matmul(new_o, y_W_o)

            # new Attn cell state
            new_state = AttentionState(new_cell_state, new_o)

            return logits, new_state


    def __call__(self, inputs, state):
        """
        Args:
            inputs: the embedding of the previous word for training only
            state: (AttentionState) (h, o) where h is the hidden state and
                o is the vector used to make the prediction of
                the previous word

        """
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)