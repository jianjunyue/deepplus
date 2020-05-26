import tensorflow as tf
import pandas as pd
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow2.Network import Network

# 在线下载汽车效能数据集

dataset_path = tf.keras.utils.get_file("auto-mpg.data",
                                       "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# 利用 pandas 读取数据集，字段有效能（公里数每加仑），气缸数，排量，马力，重量
# 加速度，型号年份，产地
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

raw_dataset.shuffle(1000)
dataset = raw_dataset.copy()
# print(dataset.head())
dataset = dataset.dropna() # 删除空白数据项
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)
dataset['Europe'] = (origin == 2)
dataset['Japan'] = (origin == 3)
# 切分为训练集和测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
# 移动 MPG 油耗效能这一列为真实标签 Y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
# 查看训练集的输入 X 的统计数据
train_stats = train_dataset.describe()
# train_stats.pop("MPG")
train_stats = train_stats.transpose()
# 标准化数据
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print(normed_train_data.shape, train_labels.shape)
print(normed_test_data.shape, test_labels.shape)

train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))  # 构建 Dataset 对象
train_db = train_db.shuffle(100).batch(32)  # 随机打散，批量化

model = Network() # 创建网络类实例
# 通过 build 函数完成内部张量的创建，其中 4 为任意的 batch 数量，9 为输入特征长度
model.build(input_shape=(4, 9))
model.summary() # 打印网络信息
optimizer = tf.keras.optimizers.RMSprop(0.001) # 创建优化器，指定学习率

# for epoch in range(200):  # 200 个 Epoch
#     for step, (x, y) in enumerate(train_db):
#         # 梯度记录器
#         with tf.GradientTape() as tape:
#             out = model(x)  # 通过网络获得输出
#             loss = tf.reduce_mean(tf.losses.MSE(y, out))  # 计算 MSE
#             mae_loss = tf.reduce_mean(tf.losses.MAE(y, out))  # 计算 MAE
#             if step % 10 == 0:  # 打印训练误差
#                 print(epoch, step, float(loss))
#             # 计算梯度，并更新
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))


model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
             loss=tf.keras.losses.mean_squared_error,
             metrics=['accuracy'])
history = model.fit(normed_train_data.values, train_labels.values, epochs=30, batch_size=32,validation_data = (normed_train_data.values, train_labels.values) )