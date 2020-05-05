import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import np
import pandas as pd
from sklearn.model_selection import train_test_split
from deepnn.lr.LinearModel import Linear

(train_data, train_label), (test_data, test_label) = tf.keras.datasets.boston_housing.load_data()
x_train_scaled, x_valid_scaled, y_train, y_valid = train_test_split(train_data, train_label, random_state = 7)
# print(x_train_scaled)

num_epochs = 5
batch_size = 50
learning_rate = 0.001

optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

model = Linear()

model.compile(loss="mse",
              optimizer=optimizer,
              metrics=['mae']
              )

# 未设置频率 默认每个epoch验证一次
history = model.fit(x_train_scaled, y_train,
                    validation_data = (x_valid_scaled, y_valid),
                    epochs = num_epochs,
                    batch_size=batch_size
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

# 预测
y_pred = model.predict(test_data)