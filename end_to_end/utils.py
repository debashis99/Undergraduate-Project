# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from third_party.positional import add_timing_signal_nd

def get_vocab(latex_vocab):
    """
    Args:
        latex_vocab : file containing vocab , one word per line

    Returns:
        dict: d[token] = id

    """
    vocab = dict()
    with open(latex_vocab) as f:
        for idx, token in enumerate(f):
            token = token.strip()
            vocab[token] = idx
        
    vocab["_UNKNOWN"] = len(vocab)
    vocab["_PAD"] = len(vocab)
    vocab["_END"] = len(vocab)
    vocab["_ST"] = len(vocab)
    
    return vocab

def get_formula_repr(formula,vocab):
    """
    Args:
        formula : a sigle latex formula 
        vocab : dict: d[token] = id

    Returns:
        a array represenation of formula

    """
    def get_token_id(token):
        return vocab[token] if token in vocab else vocab['_UNKNOWN']
    
    formula = formula.strip().split(' ')
    
    return [vocab["_ST"]] +  [get_token_id(token) for token in formula]



def pad_formula_repr(formula_repr, max_length, vocab):
    """
    Pad the the formula to max_length 
    
    """
    if(len(formula_repr) >= max_length):
        return (0, formula_repr)
    
    for i in range(len(formula_repr), max_length-1):
        formula_repr.append(vocab['_PAD'])
    
    formula_repr.append(vocab['_END'])
    
    return (1, formula_repr)
    

def greyscale(img):
    """Preprocess image (:, :, 3) image into greyscale"""
    
    img = img[:, :, 0]*0.299 + img[:, :, 1]*0.587 + img[:, :, 2]*0.114
    img = img[:, :, np.newaxis]
    return img.astype(np.uint8)   

def encode(img):
    # casting the image back to float32 on the GPU
    img = tf.cast(img, tf.float32) / 255.

    out = tf.layers.conv2d(img, 64, 3, 1, "SAME", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

    out = tf.layers.conv2d(out, 128, 3, 1, "SAME", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

    out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)

    out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

    out = tf.layers.conv2d(out, 512, 3, 1, "SAME", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")

    # encoder representation, shape = (batch size, height', width', 512)
    out = tf.layers.conv2d(out, 512, 3, 1, "VALID", activation=tf.nn.relu)

    out = add_timing_signal_nd(out)

    size = tf.shape(out)
    seq = tf.reshape(out, shape=[-1, size[1]*size[2], 512])

    return seq


def pad_batch_formulas(formulas, id_pad, id_end, max_len=None):
    """Pad formulas to the max length with id_pad and adds and id_end token
    at the end of each formula
    Args:
        formulas: (list) of list of ints
        max_length: length maximal of formulas
    Returns:
        array: of shape = (batch_size, max_len) of type np.int32
        array: of shape = (batch_size) of type np.int32
    """
    if max_len is None:
        max_len = max(map(lambda x: len(x), formulas))

    batch_formulas = id_pad * np.ones([len(formulas), max_len+1],
            dtype=np.int32)
    formula_length = np.zeros(len(formulas), dtype=np.int32)
    for idx, formula in enumerate(formulas):
        batch_formulas[idx, :len(formula)] = np.asarray(formula,
                dtype=np.int32)
        batch_formulas[idx, len(formula)]  = id_end
        formula_length[idx] = len(formula) + 1

    return batch_formulas, formula_length
