from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

path="C:\\Users\\lejianjun\\git\\deepplus\\data\\"
train_file =path+ 'HanXiaoyang\\data\\'

def readPath(file):
    return train_file+file

# 读取数据，统计基本的信息，field等
DTYPE = tf.float32

FIELD_SIZES = [0] * 26
with open(readPath("featindex.txt")) as fin:
    for line in fin:
        line = line.strip().split(':')
        if len(line) > 1:
            f = int(line[0]) - 1
            FIELD_SIZES[f] += 1
print('field sizes:', FIELD_SIZES)
FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = sum(FIELD_SIZES)
OUTPUT_DIM = 1
STDDEV = 1e-3
MINVAL = -1e-3
MAXVAL = 1e-3

def read_data(file_name):
    X = []
    D = []
    y = []
    with open(file_name) as fin:
        for line in fin:
            fields = line.strip().split()
            y_i = int(fields[0])
            X_i = [int(x.split(':')[0]) for x in fields[1:]]
            D_i = [int(x.split(':')[1]) for x in fields[1:]]
            y.append(y_i)
            X.append(X_i)
            D.append(D_i)
    y = np.reshape(np.array(y), [-1])
    X = libsvm_2_coo(zip(X, D), (len(X), INPUT_DIM)).tocsr()
    return X, y

# 工具函数，libsvm格式转成coo稀疏存储格式
def libsvm_2_coo(libsvm_data, shape):
    coo_rows = []
    coo_cols = []
    coo_data = []
    n = 0
    for x, d in libsvm_data:
        coo_rows.extend([n] * len(x))
        coo_cols.extend(x)
        coo_data.extend(d)
        n += 1
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.array(coo_data)
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)

# 读取数据
train_data = read_data(readPath("train.txt"))
test_data = read_data(readPath("test.txt"))

print(type(train_data))
print(train_data)


print('train data size:', train_data[0].shape)
print('test data size:', test_data[0].shape)
