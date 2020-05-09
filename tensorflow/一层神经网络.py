import tensorflow as tf

# y=W1@x + b1

y=tf.ones([60000,10])
print(y.shape)
x=tf.random.normal([60000,784])

W1=tf.Variable(tf.random.normal([784,10]))
b1=tf.Variable(tf.zeros([10]))
# print(W1.shape)
# print(b1.shape)
print(b1)
# print("--------y1-----------")
# y1=x@W1+b1
# print(y1.shape)

lr=0.001
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = x@W1+b1
        loss = tf.square(y - y_pred) # sum(y-out)^2
        loss = tf.reduce_mean(loss) # 计算网络输出与标签之间的均方差， mse = mean(sum(y-out)^2)
    grads = tape.gradient(loss, [W1, b1]) # 自动梯度，需要求梯度的张量有[w1, b1, w2, b2, w3, b3]

    # 梯度更新， assign_sub 将当前值减去参数值，原地更新
    W1.assign_sub(lr * grads[0])
    b1.assign_sub(lr * grads[1])

    print('loss:', loss.numpy())
    print(b1)
