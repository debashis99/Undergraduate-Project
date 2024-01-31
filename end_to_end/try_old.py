
import tensorflow as tf
from utils import greyscale, get_vocab, encode, pad_batch_formulas, get_formula_repr
import numpy as np
import imageio,time
from third_party.positional import add_timing_signal_nd

import collections

from tensorflow.contrib.rnn import RNNCell, LSTMCell, LSTMStateTuple


_dir_data = 'data/sample/'

i_img = imageio.imread(_dir_data + "/" + 'images_processed/1a6ad5d0f5.png') # a numpy array
i_img = greyscale(i_img)

vocab = get_vocab(_dir_data + "/" + 'latex_vocab.txt')

vocab_size = len(vocab)

train_set = []
with open(_dir_data + 'train_filter.lst') as my_file:
    for line in my_file:
        line = line.strip().split(' ')
        train_set.append((line[0],line[1]))
        


print("HI BRO")
#####################################################################################33
# batch of images, shape = (batch size, height, width, 1)
img = tf.placeholder(tf.uint8, shape=(None, None, None, 1), name='img')

# batch of formulas, shape = (batch size, length of the formula)
formula = tf.placeholder(tf.int32, shape=(None, None), name='formula')
# for padding
formula_length = tf.placeholder(tf.int32, shape=(None, ), name='formula_length')

seq = encode(img)

n = tf.shape(seq)[1]


# learn the first hidden vector of LSTM

img_mean = tf.reduce_mean(seq, axis=1)
W = tf.get_variable("W", shape=[512, 512])
b = tf.get_variable("b", shape=[512])
h = tf.tanh(tf.matmul(img_mean, W) + b)


def get_attention_vector(seq,h):
    # over the image, shape = (batch size, n, 512)
    W1_e = tf.layers.dense(inputs=seq, units=512, use_bias=False)
    # over the hidden vector, shape = (batch size, 512)
    W2_h = tf.layers.dense(inputs=h, units=512, use_bias=False)

    # sums the two contributions
    a = tf.tanh(W1_e + tf.expand_dims(W2_h, axis=1))
    beta = tf.get_variable("beta", shape=[512, 1], dtype=tf.float32)
    a_flat = tf.reshape(a, shape=[-1, 512])
    a_flat = tf.matmul(a_flat, beta)
    a = tf.reshape(a, shape=[-1, n])

    # compute weights
    a = tf.nn.softmax(a)
    a = tf.expand_dims(a, axis=-1)
    c = tf.reduce_sum(a * seq, axis=1)

    W3_o = tf.layers.dense(inputs=tf.concat([h, c], axis=-1), units=512, use_bias=False)
    o = tf.tanh(W3_o)

    return o

h_0 = h
o_0 = get_attention_vector(seq,h_0)

## our attention cell 
AttentionHiddenState = collections.namedtuple("AttentionHiddenState", ("lstm_state", "o"))

class AttentionCell(RNNCell):
    def __init__(self):
        self.lstm_cell = LSTMCell(512)
        # # variables and tensors
        # self._cell                = LSTMCell(512)
        

        # hyperparameters and shapes
        self._n_channels     = n
        self._dim_e          = 512
        self._dim_o          = 512
        self._num_units      = 512
        self._dim_embeddings = 80
        self._num_proj       = vocab_size
        self._dtype          = tf.float32

        # for RNNCell
        self._state_size = AttentionHiddenState(self.lstm_cell._state_size, self._dim_o)


    @property
    def state_size(self):
        return self._state_size


    @property
    def output_size(self):
        return self._num_proj


    @property
    def output_dtype(self):
        return self._dtype


    def __call__(self, inputs, cell_state):
        """
        Args:
            inputs: shape = (batch_size, dim_embeddings) embeddings from previous time step
            cell_state: (AttentionState) state from previous time step
        """
        lstm_state, o = cell_state

        # compute h
        h, new_lstm_state = self.lstm_cell(tf.concat([inputs, o], axis=-1), lstm_state)
       
        new_o  = get_attention_vector(h,seq)

        logits = tf.layers.dense(inputs=new_o, units=vocab_size, use_bias=False)

        new_state = AttentionHiddenState(new_lstm_state, new_o)
        return logits, new_state


def get_embeddings(formula, E, dim, start_token, batch_size):
    """Returns the embedding of the n-1 first elements in the formula concat
    with the start token
    Args:
        formula: (tf.placeholder) tf.uint32
        E: tf.Variable (matrix)
        dim: (int) dimension of embeddings
        start_token: tf.Variable
        batch_size: tf variable extracted from placeholder
    Returns:
        embeddings_train: tensor
    """
    formula_ = tf.nn.embedding_lookup(E, formula)
    start_token_ = tf.reshape(start_token, [1, 1, dim])
    start_tokens = tf.tile(start_token_, multiples=[batch_size, 1, 1])
    embeddings = tf.concat([start_tokens, formula_[:, :-1, :]], axis=1)

    return embeddings


def embedding_initializer():
    """Returns initializer for embeddings"""
    def _initializer(shape, dtype, partition_info=None):
        E = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        E = tf.nn.l2_normalize(E, -1)
        return E

    return _initializer

dim_embeddings = 80
E = tf.get_variable("E", initializer=embedding_initializer(),
    shape=[vocab_size, dim_embeddings], dtype=tf.float32)

start_token = tf.get_variable("start_token", dtype=tf.float32,
    shape=[dim_embeddings], initializer=embedding_initializer())

batch_size = tf.shape(img)[0]

embeddings = get_embeddings(formula, E, dim_embeddings, start_token, batch_size)



# 3. decode
attn_cell = AttentionCell()

seq_logits, _ = tf.nn.dynamic_rnn(attn_cell, embeddings, initial_state=AttentionHiddenState(h_0, o_0))



# compute - log(p_i[y_i]) for each time step, shape = (batch_size, formula length)
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seq_logits, labels=formula)
# masking the losses
mask = tf.sequence_mask(formula_length)
losses = tf.boolean_mask(losses, mask)
# averaging the loss over the batch
loss = tf.reduce_mean(losses)
# building the train op
optimizer = tf.train.AdamOptimizer(0.1)
train_op = optimizer.minimize(loss)



#init variable --> always keep at last
init = tf.global_variables_initializer()


def run_epoch( train_set, epoch ,sess,train_op,loss):
    """Performs an epoch of training
    Args:
        
        train_set: Dataset instance
       
        epoch: (int) id of the epoch, starting at 0
    Returns:
        score: (float) model will select weights that achieve the highest
            score
    """
    # logging
    batch_size = 5
    
    def batches(train_set, batch_size):
        x_batch, y_batch = [], []
        for (x, y) in train_set:
            if len(x_batch) == batch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
            
            i_img = imageio.imread(_dir_data  + 'images_processed/' + str(x)) # a numpy array
            x = greyscale(i_img)

            y = ""
            with open(_dir_data  + 'formula_processed/' + str(y), 'r') as myfile:
                y=myfile.read().replace('\n', '')
            
            y = get_formula_repr(y,vocab)

            x_batch += [x]
            y_batch += [y]

            

        if len(x_batch) != 0:
            yield x_batch, y_batch

  
    # iterate over dataset
    for i, (img, formula) in enumerate(batches(train_set, batch_size)):
        # get feed dict
        formula, formula_length = pad_batch_formulas(formula, vocab["_PAD"], vocab["_END"])
        fd = {
                img :img,
                formula : formula,
                formula_length : formula_length
            }
        # update step
        _, loss_eval = sess.run([train_op,loss], feed_dict=fd)
        print("SINGLE BATCH: ", loss_eval)

    return 1

# #launch it 
# with tf.Session() as sess:
#     print("HI FROM SESSION")
#     sess.run(init)

#     best_score = None

#     for epoch in range(100):
#         # logging
#         tic = time.time()

#         # epoch
#         score = run_epoch(train_set, epoch,sess,train_op,loss)

#         # save weights if we have new best score on eval
#         if best_score is None or score >= best_score:
#             best_score = score
        
    
#         toc = time.time()

#         print("SINGLE EPOCH: ", epoch)
    

    
