import tensorflow as tf
from pandas import np

from deepctrplus.layers import LayerNormalization

# 深度学习中Embedding层
# https://blog.csdn.net/u010412858/article/details/77848878

model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(2, 2, input_length=7))
model.compile('rmsprop', 'mse')
a=model.predict(  [[0,1,0,1,1,0,0]])

print(a)