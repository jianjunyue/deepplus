
import tensorflow as tf
from tensorflow.python.keras import backend as K

class FMLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):

        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)

