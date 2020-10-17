import tensorflow as tf
import  tensorflow.keras as keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, left, right):
        super(SliceLayer, self).__init__()
        self.left = left
        self.right = right

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return inputs[:, self.left:self.right]

'''
 参数:
 args: 自定义构建参数
'''
def model_build():
    inputs = tf.keras.layers.Input((39,))
    wide = SliceLayer(10, 30)(inputs)
    userid = SliceLayer(1, 2)(inputs)
    vehicle = SliceLayer(2, 3)(inputs)
    city = SliceLayer(3, 4)(inputs)
    prd = SliceLayer(4, 5)(inputs)
    user_emb = keras.layers.Embedding(input_dim=1000000, output_dim=5, input_length=1)(userid)
    vehicle_emb = keras.layers.Embedding(input_dim=30000, output_dim=3, input_length=1)(vehicle)
    city_emb = keras.layers.Embedding(input_dim=1000, output_dim=2, input_length=1)(city)
    prd_emb = keras.layers.Embedding(input_dim=10000, output_dim=3, input_length=1)(prd)
    flat1 = keras.layers.Flatten()(user_emb)
    flat2 = keras.layers.Flatten()(vehicle_emb)
    flat3 = keras.layers.Flatten()(city_emb)
    flat4 = keras.layers.Flatten()(prd_emb)
    concat = keras.layers.concatenate([flat1, flat2, flat3, flat4, inputs])
    dense = keras.layers.Dense(32, activation='relu')(concat)
    output = keras.layers.Dense(1, activation='sigmoid')(dense)
    model = keras.models.Model(inputs, output)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.summary()
    return model

model=model_build()
data = pd.read_csv('C:\\Users\\lejianjun\\PycharmProjects\\deepplus\\data\\criteo_sample.txt')
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

train, test = train_test_split(data, test_size=0.2, random_state=2020)

history = model.fit(train[sparse_features+dense_features], train[target].values,batch_size=256, epochs=10, verbose=2, validation_split=0.2 )

pred_ans = model.predict(test[sparse_features+dense_features], batch_size=256)
print(pred_ans)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))