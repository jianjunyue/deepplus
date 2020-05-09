import tensorflow as tf


x = tf.random.normal([10,35,8])

result = tf.split(x,axis=0,num_or_size_splits=[4,2,2,2])
#
# print(result[0].shape)
#
# result = tf.unstack(x,axis=0) # Unstack 为长度为 1
# print(len(result))
# print(result[0].shape)
#
# x = tf.ones([2,2])
# print(x)
# print(tf.norm(x,ord=1))
# print(tf.norm(x,ord=2))
#
# out = tf.random.normal([4,10]) # 网络预测输出
# y = tf.constant([1,2,2,0]) # 真实标签
# y = tf.one_hot(y,depth=10) # one-hot 编码
# loss = tf.keras.losses.mse(y,out) # 计算每个样本的误差
# loss1 = tf.reduce_mean(loss) # 平均误差
# print(loss)
# print(loss1)

# out = tf.random.normal([2,10])
# max=tf.reduce_max(out, axis=1) # 返回第一维概率最大的值
# pred = tf.argmax(out, axis=1) # 返回第一维概率最大值的位置
# argmin = tf.argmin(out, axis=1) # 返回第一维概率最小值的位置
# print(out)
# print(max)
# print(pred)
# print(argmin)
# a=tf.random.uniform([2,5],maxval=5,dtype=tf.int32)
# b=tf.random.uniform([2,5],maxval=5,dtype=tf.int32)
#
# c=tf.equal(a,b)
# d = tf.cast(c, dtype=tf.float32) # 布尔型转 int 型
# e=tf.reduce_sum(d,axis=1)
# f=tf.reduce_sum(d)
# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# print(f)
# g=tf.math.greater(a,b)
# print("g",g)
#
# h=tf.math.less(a,b)
# print("h",h)
#
# m=tf.math.greater_equal(a,b)
# print("m",m)
#
# n=tf.math.less_equal(a,b)
# print("n",n)
#
# l=tf.math.not_equal(a,b)
# print("l",l)
#
# t=tf.math.is_nan(a)
# print("t",t)

# b = tf.constant([1,2,3,4])
# print(b.shape)
# #
# shapep = tf.constant([[1,3]])
# print(p.shape)
# # c = tf.pad(b, p) # 填充
# # print(c)
#
# b = tf.constant([[1, 2, 3, 4]])
# print(b.shape)
# p = tf.constant([[1,3],[1,2]])
# print(p.shape)
# c = tf.pad(b, p) # 填充
# print(c)

print("---------------")
b = tf.constant([[1,3],[1,2]])
print(b.shape)
p = tf.constant([[1,3],[1,2]])
print(p.shape)
c = tf.pad(b, p) # 填充
print(c)
# print("---------------")
# b = tf.constant([[1,2],[3,4]]) # 填充
# print(b.shape)
# p = tf.constant([[1,3],[1,2]])
# print(p.shape)
# c = tf.pad(b, p) # 填充
# print(c)