import tensorflow as tf

def load_data():
    # 加载 MNIST 数据集
    (x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    # 转换为浮点张量， 并缩放到-1~1
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
    # 转换为整形张量
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    # one-hot 编码
    y = tf.one_hot(y, depth=10)

    print(y.shape)

    # 改变视图， [b, 28, 28] => [b, 28*28]
    x = tf.reshape(x, (-1, 28 * 28))
    print(x.shape)

    print("-----------------x--y----------")

    # 构建数据集对象
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # 批量训练
    train_dataset = train_dataset.batch(200)
    return train_dataset

load_data()


