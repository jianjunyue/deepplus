import tensorflow as tf


class LossUtils:

    """
    平均差损失-->mean_squared_error
    """
    def MeanSquaredError(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))