import tensorflow as tf

# https://github.com/chengstone/kaggle_criteo_ctr_challenge-/blob/master/ctr_tf2.ipynb

class CTRNetwork(tf.keras.models.Model):
    def __init__(self,sparse_max,embed_dim,sparse_dim,out_dim):
        super(CTRNetwork, self).__init__()

        # ffm_fc_layer = tf.keras.layers.Dense(1, name="ffm_fc_layer")  # FFM_input
        # fm_fc_layer = tf.keras.layers.Dense(1, name="fm_fc_layer")  # FM_input
        # 输入类别特征，从嵌入层获得嵌入向量
        # input_dim：大或等于0的整数，字典长度，即输入数据最大下标 + 1
        # output_dim：大于0的整数，代表全连接嵌入的(Embedding)维度
        # input_length：(输入序列的维度 )当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
        self.sparse_temp_embed_layer = tf.keras.layers.Embedding(input_dim=sparse_max, output_dim=embed_dim, input_length=sparse_dim)
        self.sparse_embed_layer = tf.keras.layers.Reshape([sparse_dim * embed_dim])

        self.fc_layer1=tf.keras.layers.Dense(out_dim, name="fc_layer1", activation='relu')
        self.fc_layer2=tf.keras.layers.Dense(out_dim, name="fc_layer2", activation='relu')
        self.fc_layer3=tf.keras.layers.Dense(out_dim, name="fc_layer3", activation='relu')

        self.ffm_fc_layer = tf.keras.layers.Dense(1, name="ffm_fc_layer")
        self.fm_fc_layer = tf.keras.layers.Dense(1, name="fm_fc_layer")

        self.logits_output_layer = tf.keras.layers.Dense(1, name="logits_layer", activation='sigmoid')

    def call(self, input):
        dense_input,sparse_input=input
        FFM_input,FM_input=input

        print("dense_input.shape :"+str(dense_input.shape))
        print("sparse_input.shape :"+str(sparse_input.shape))
        sparse_temp_embed_layer=self.sparse_temp_embed_layer(sparse_input)
        print("sparse_temp_embed_layer.shape :"+str(sparse_temp_embed_layer.shape))
        sparse_embed_layer=self.sparse_embed_layer(sparse_temp_embed_layer)
        print("sparse_embed_layer.shape :"+str(sparse_embed_layer.shape))

        # 输入数值特征，和嵌入向量链接在一起经过三层全连接层
        input_combine_layer = tf.keras.layers.concatenate([dense_input, sparse_embed_layer])
        print("input_combine_layer.shape :"+str(input_combine_layer.shape))
        fc1_layer=self.fc_layer1(input_combine_layer)
        print("fc1_layer.shape :"+str(fc1_layer.shape))
        fc2_layer=self.fc_layer2(fc1_layer)
        print("fc2_layer.shape :"+str(fc2_layer.shape))
        fc3_layer=self.fc_layer3(fc2_layer)
        print("fc3_layer.shape :"+str(fc3_layer.shape))

        ffm_fc_layer=self.ffm_fc_layer(FFM_input)
        print("ffm_fc_layer.shape :"+str(ffm_fc_layer.shape))
        fm_fc_layer=self.fm_fc_layer(FM_input)
        print("fm_fc_layer.shape :"+str(fm_fc_layer.shape))

        feature_combine_layer = tf.keras.layers.concatenate([ffm_fc_layer, fm_fc_layer, fc3_layer], 1)
        print("feature_combine_layer.shape :"+str(feature_combine_layer.shape))

        logits_output=self.logits_output_layer(feature_combine_layer)
        print("logits_output.shape :"+str(logits_output.shape))

        return logits_output