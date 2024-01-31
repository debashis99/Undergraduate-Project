import tensorflow as tf


class AttentionMechanism(object):
    """Class to compute attention over an image"""

    def __init__(self, img, tiles=1):
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


    def context(self, h, tiles,regions, _img, _attn_img, ):
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


