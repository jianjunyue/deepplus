import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import np
import pandas as pd
from sklearn.model_selection import train_test_split
from deepnn.mlp.MLP import MLP

(train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()
#
# # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
test_data = np.expand_dims(test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
train_label = train_label.astype(np.int32)  # [60000]
test_label = test_label.astype(np.int32)  # [10000]
#
x_train_scaled, x_valid_scaled, y_train, y_valid = train_test_split(train_data, train_label, random_state = 7)
# print(x_train_scaled)

num_epochs = 5
batch_size = 50
learning_rate = 0.001

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model = MLP()

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy']
              )

# 未设置频率 默认每个epoch验证一次
history = model.fit(x_train_scaled, y_train,
                    validation_data = (x_valid_scaled, y_valid),
                    epochs = 2
                    )

# 画学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
plot_learning_curves(history)



# 测试
loss,accuracy = model.evaluate(x_valid_scaled, y_valid)
print('\ntest loss',loss)
print('accuracy',accuracy)





