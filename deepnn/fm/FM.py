import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import np
import pandas as pd

class FM_Model(tf.keras.Model):

    def __init__(self, num_feat, num_field, reg_l1=0.01, reg_l2=0.01, embedding_size=10):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2  # L1/L2正则化并没有去使用
        self.num_feat = num_feat  # denote as M
        self.num_field = num_field  # denote as F
        self.embedding_size = embedding_size  # denote as K

        # first order term parameters embedding
        self.first_weights = tf.keras.layers.Embedding(num_feat, 1, embeddings_initializer='uniform')
        self.bias = tf.Variable([0.0])

        self.feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size, embeddings_initializer='uniform')

    def call(self, feat_index, feat_value):
        # Step1: 先计算得到线性的那一部分
        feat_value = tf.expand_dims(feat_value, axis=-1)
        first_weights = self.first_weights(feat_index)
        first_weight_value = tf.math.multiply(first_weights, feat_value)
        first_weight_value = tf.squeeze(first_weight_value, axis=-1)
        y_first_order = tf.math.reduce_sum(first_weight_value, axis=1)

        # Step2: 再计算二阶部分
        secd_feat_emb = self.feat_embeddings(feat_index)                        # None * F * K
        feat_emd_value = tf.math.multiply(secd_feat_emb, feat_value)            # None * F * K(广播)

        # sum_square part
        summed_feat_emb = tf.math.reduce_sum(feat_emd_value, axis=1)            # None * K
        interaction_part1 = tf.math.pow(summed_feat_emb, 2)                     # None * K

        # squared_sum part
        squared_feat_emd_value = tf.math.pow(feat_emd_value, 2)                 # None * K
        interaction_part2 = tf.math.reduce_sum(squared_feat_emd_value, axis=1)  # None * K

        y_secd_order = 0.5 * tf.subtract(interaction_part1, interaction_part2)
        y_secd_order = tf.math.reduce_sum(y_secd_order, axis=1)

        output = y_first_order + y_secd_order + self.bias
        output = tf.expand_dims(output, axis=1)
        return output