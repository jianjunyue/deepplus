
from tensorflow import keras
import tensorflow as tf

# 自定义DenseLyer

class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

     #初始化参数
    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        """构建所需要的参数"""
        # x*w+b  input_shape:[None,a] w:[a,b] output_shape:[None,b]

        self.w = self.add_weight(name='w',
                                      shape=[input_shape[1], self.units],
                                      initializer='uniform',
                                      trainable=True)
        self.b = self.add_weight(name='b',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        """完成正向计算"""
        return self.activation(x @ self.w + self.b)