import tensorflow as tf

#  y = 𝑟𝑒𝑙𝑢{𝑟𝑒𝑙𝑢{𝑟𝑒𝑙𝑢[𝑋@W1 + b1]@W2 +b2}@W3 +b3}

y = tf.random.normal([60000, 10])
print(y.shape)
x = tf.random.normal([60000, 784])

W1 = tf.Variable(tf.random.normal([784, 256]))
b1 = tf.Variable(tf.zeros([256]))
W2 = tf.Variable(tf.random.normal([256, 128]))
b2 = tf.Variable(tf.zeros([128]))
W3 = tf.Variable(tf.random.normal([128, 10]))
b3 = tf.Variable(tf.zeros([10]))

lr = 0.001
for i in range(10):
    with tf.GradientTape() as tape:
        # 第一层计算， [60000, 784]@[784, 256] + [256] => [60000, 256] + [256] => [60000,256] + [60000, 256]=> [60000,256]
        h1 = x @ W1 + b1
        h1 = tf.nn.relu(h1)  # 通过激活函数

        # 第二层计算， [60000, 256]@[256, 128] + [128] => [60000, 128]
        h2 = h1 @ W2 + b2
        h2 = tf.nn.relu(h2)

        # 第二层计算，  [60000, 128]@[128, 10] + [10] => [60000, 10]
        h3 = h2 @ W3 + b3
        y_pred = h3
        loss = tf.square(y - y_pred)  # sum(y-out)^2
        loss = tf.reduce_mean(loss)  # 计算网络输出与标签之间的均方差， mse = mean(sum(y-out)^2)
    grads = tape.gradient(loss, [W1, b1,W2, b2,W3, b3])  # 自动梯度，需要求梯度的张量有[w1, b1, w2, b2, w3, b3]

    # 梯度更新， assign_sub 将当前值减去参数值，原地更新
    W1.assign_sub(lr * grads[0])
    b1.assign_sub(lr * grads[1])
    W2.assign_sub(lr * grads[2])
    b2.assign_sub(lr * grads[3])
    W3.assign_sub(lr * grads[4])
    b3.assign_sub(lr * grads[5])

    print(i, 'loss:', loss.numpy())
    # print(b1)
