import tensorflow as tf

# x = tf.linspace(-5.0,5,11)
# print(type(x))
# print(x)
#
# data=tf.keras.datasets.boston_housing.load_data()
#
# data=data.shuffle(1000)
#
# train_db = data.batch(128)
# print(type(train_db))

# 创建网络参数 w1,w2
w1 = tf.random.normal([4,3])
w2 = tf.random.normal([4,2])
print(tf.math.abs(w1))
print(tf.reduce_sum(tf.math.abs(w1)))
# 计算 L1 正则化项
loss_reg = tf.reduce_sum(tf.math.abs(w1)) + tf.reduce_sum(tf.math.abs(w2))
print(loss_reg)

# 计算 L2 正则化项
loss_reg = tf.reduce_sum(tf.square(w1)) + tf.reduce_sum(tf.square(w2))
print(loss_reg)