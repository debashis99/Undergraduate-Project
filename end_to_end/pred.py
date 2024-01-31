
import tensorflow as tf
from utils import greyscale, get_vocab, encode, pad_batch_formulas, get_formula_repr
import numpy as np
import imageio,time
from third_party.positional import add_timing_signal_nd

from lstm_with_attention import LstmAttentionCell

from components.dynamic_decode import dynamic_decode
from components.attention_mechanism import AttentionMechanism
from components.attention_cell import AttentionCell
from components.beam_search_decoder_cell import BeamSearchDecoderCell
import sys


tf.reset_default_graph()

_dir_data = 'data/sample/'


vocab = get_vocab(_dir_data + "/" + 'latex_vocab.txt')

vocab_size = len(vocab)

print(vocab["_ST"], vocab["_END"])

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

img = tf.cast(img, tf.float32) / 255

seq = encode(img)

n = tf.shape(seq)[1]


# learn the first hidden vector of LSTM

img_mean = tf.reduce_mean(seq, axis=1)
W = tf.get_variable("W", shape=[512, 512])
b = tf.get_variable("b", shape=[512])
h = tf.tanh(tf.matmul(img_mean, W) + b)


def get_embeddings(formula, E, dim, batch_size):
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
    embeddings = tf.nn.embedding_lookup(E, formula) # [len_formula, 80]
    
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

batch_size = tf.shape(img)[0]


# training
with tf.variable_scope("attn_cell", reuse=False):
    embeddings = get_embeddings(formula, E, dim_embeddings, batch_size)
    recu_cell =  tf.contrib.rnn.LSTMCell(512)
    attn_cell = AttentionCell(recu_cell, vocab_size, seq)

    seq_logits, _ = tf.nn.dynamic_rnn(attn_cell, embeddings, initial_state=attn_cell.initial_state())

# decoding
with tf.variable_scope("attn_cell", reuse=True):
    recu_cell =  tf.contrib.rnn.LSTMCell(512,reuse = True)
    attn_cell = AttentionCell(recu_cell, vocab_size, seq, tiles = 2)
    
    # decoder_cell = GreedyDecoderCell(E, attn_cell, batch_size, tf.nn.embedding_lookup(E, vocab["_ST"]) , vocab["_END"])
   
    decoder_cell = BeamSearchDecoderCell(E, attn_cell, batch_size,
                tf.nn.embedding_lookup(E, vocab["_ST"]), vocab["_END"], 2, 1, 0)

    test_outputs, _ = dynamic_decode(decoder_cell,101)




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



# Add ops to save and restore all the variables.
saver = tf.train.Saver()


#launch it 
with tf.Session() as sess:
    print("HI FROM SESSION")
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")

   
    for (x,y ) in train_set:

        print(x, y)

        x = imageio.imread(_dir_data  + 'images_processed/' + str(x)) # a numpy array
        x = greyscale(x)
        with open(_dir_data  + 'formula_processed/' + str(y) + ".txt", 'r') as myfile:
            y=myfile.read().replace('\n', '')
        y = get_formula_repr(y,vocab)
        i_img = [x]                
        formul = [y]
        formul, formul_len = pad_batch_formulas(formul, vocab["_PAD"], vocab["_END"])
        fd = { img :i_img, formula : formul, formula_length :formul_len }
        pred = sess.run(test_outputs,feed_dict=fd )[1]

        
        print(len(pred) , pred)

        print('\n\n')
    

            
        
            
       
    

    
