
from tensorflow import keras
import tensorflow as tf

# 方式二子类API
# 子类API


class mv_network_model(keras.models.Model):
    def __init__(self):
        super(mv_network_model, self).__init__()
        self.input_dim=50

        self.user_job_embed_layer = tf.keras.layers.Embedding(input_dim=21,output_dim= 16, input_length=1, name='user_job_embed_layer')
        self.user_gender_embed_layer = tf.keras.layers.Embedding(input_dim=2,output_dim= 16,input_length=1, name='user_gender_embed_layer')

        self.user_job_layer = tf.keras.layers.Dense(32, name="user_job_layer_32", activation='relu')
        self.user_gender_layer = tf.keras.layers.Dense(32, name="user_gender_laye_32r", activation='relu')

        self.user_combine_layer = tf.keras.layers.Dense(200, activation='tanh') # (?, 1, 200)

        self.user_combine_layer_flat = tf.keras.layers.Reshape([200], name="user_combine_layer_flat")

        self.movie_id_embed_layer = tf.keras.layers.Embedding(3953, 32, input_length=1,   name='movie_id_embed_layer')
        self.movie_categories_embed_layer = tf.keras.layers.Embedding(19, 32, input_length=19, name='movie_categories_embed_layer')

        self.movie_id_fc_layer = tf.keras.layers.Dense(32, name="movie_id_fc_layer", activation='relu')
        self.movie_categories_fc_layer = tf.keras.layers.Dense(32, name="movie_categories_fc_layer", activation='relu')

        self.movie_combine_layer = tf.keras.layers.Dense(200, activation='tanh') # (?, 1, 200)

        self.inference_combine_layer = tf.keras.layers.Dense(64, activation='tanh') # (?, 1, 200)

        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input):
        split0, split1, split2, split3, split4=input
        layer1 = self.user_job_embed_layer(split4 )
        layer2 = self.user_job_layer(layer1)
        #
        layer3 = self.user_gender_embed_layer(split2)
        layer4 = self.user_gender_layer(layer3)
        #
        layer5= tf.keras.layers.concatenate([layer2, layer4])  # (?, 1, 128)

        layer6 = self.user_combine_layer(layer5)
        #
        layer7 = self.movie_id_embed_layer(split1)
        layer8 = self.movie_id_fc_layer(layer7)
        #
        # layer9 = self.movie_categories_embed_layer(split5)
        layer9 = self.user_gender_embed_layer(split2) # 临时 user_gender
        layer10 = self.movie_categories_fc_layer(layer9)
        #
        layer11= tf.keras.layers.concatenate([layer8, layer10])  # (?, 1, 128)
        layer12 = self.movie_combine_layer(layer11)
        #
        # # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
        inference_layer = tf.keras.layers.concatenate([layer6, layer12])  # (?, 400)
        # # 你可以使用下面这个全连接层，试试效果
        inference_dense = self.inference_combine_layer(inference_layer)
        output = self.output_layer(inference_dense)
        return output