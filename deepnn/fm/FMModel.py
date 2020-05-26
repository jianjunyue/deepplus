import tensorflow as tf

from deepnn.fm.FMLayer import FMLayer

class FMModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.dense = tf.keras.layers.Dense(
        #     units=1,
        #     activation=None,
        #     kernel_initializer=tf.zeros_initializer(),
        #     bias_initializer=tf.zeros_initializer()
        # )
        self.layer = FMLayer(units=1)
        # self.layer = CustomizedDenseLayer(units=1,activation="relu")


    def call(self, inputs):
        # output = self.dense(inputs)
        output = self.layer(inputs)
        return output