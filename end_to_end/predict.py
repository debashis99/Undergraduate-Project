import tensorflow as tf 
import numpy as np
import imageio

dir_data = 'data/sample/'
dir_model = 'model/sample'

vocab = get_vocab(_dir_data + "/" + 'latex_vocab.txt')
vocab_size = len(vocab)


tf.reset_default_graph()


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
 
 
    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data
 
    graph = tf.get_default_graph()
    img = graph.get_tensor_by_name("img:0")
    formula = graph.get_tensor_by_name("formula:0")
    formula_length = graph.get_tensor_by_name("formula_length:0")

    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
      embedding=embedding,
      start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
      end_token=END_SYMBOL)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cell,
        helper=helper,
        initial_state=cell.zero_state(batch_size, tf.float32))
    outputs, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=20)

    print sess.run(op_to_restore,feed_dict)



  decoder_cell = tf.contrib.seq2seq.BeamSearchDecoder( cell = attn_cell,
                                                       embedding = E,
                                                       start_tokens = tf.tile(vocab["_ST"],[batch_size]),
                                                       end_token = vocab["_END"],
                                                       initial_state = AttentionState(
                                                            tf.contrib.seq2seq.tile_batch(attn_cell.initial_state()[0], multiplier=5),
                                                            tf.contrib.seq2seq.tile_batch(attn_cell.initial_state()[1], multiplier=5)),
                                                       beam_width = 5
                                                    )
    
    


    test_outputs, _ = tf.contrib.seq2seq.dynamic_decode( decoder=decoder_cell,
        output_time_major=False,
        impute_finished=False,
        maximum_iterations=101)