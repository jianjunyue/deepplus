import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import sklearn
import time
import sys
import os

from deepnn.widedeep.WideDeepModel import WideDeepModel

housing = fetch_california_housing()
print(housing.data.shape)   # (20640, 8)
print(housing.target.shape) # (20640, )

# 切分数据集
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state = 7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state = 11)
print(x_train.shape, y_train.shape) # (11610, 8) (11610,)
print(x_valid.shape, y_valid.shape) # (3870, 8) (3870,)
print(x_test.shape, y_test.shape)   # (5160, 8) (5160,)

# 数据归一化 x = (x - u) / d
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

model = WideDeepModel()
model.build(input_shape=(None, 8))

model.compile(loss="mean_squared_error",
              optimizer="sgd"
              )

# 未设置频率 默认每个epoch验证一次
history = model.fit(x_train_scaled, y_train,
                    validation_data = (x_valid_scaled, y_valid),
                    epochs = 100
                    )

# 画学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
plot_learning_curves(history)

# 测试
model.evaluate(x_test_scaled, y_test)