import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def data_preparing(filepath, n_samples=50000):
    # data = []
    #     # n = 1
    #     # progress = tqdm(total=n_samples)
    #     # with open(filepath, "r") as f:
    #     #     for line in f:
    #     #         if n > n_samples:
    #     #             break
    #     #         line = line.strip()
    #     #         line_data = line.split("\t")
    #     #         line_data = [None if x == "" else x for x in line_data]
    #     #         data.append(line_data)
    #     #         progress.update(1)
    #     #         n += 1
    #     # progress.close()
    #     # df = pd.DataFrame(data)
    #     # integer_cols = [f"I{i}" for i in range(13)]
    #     # categorical_cols = [f"C{i}" for i in range(26)]
    #     # df.columns = ["label"] + integer_cols + categorical_cols
    df = pd.read_csv(filepath)
    sparse_cols = ['C' + str(i) for i in range(1, 27)] # 类型
    dense_cols = ['I' + str(i) for i in range(1, 14)] #值
    return df, dense_cols, sparse_cols


def data_process(df, dense_cols, sparse_cols):
    for col in dense_cols:
        df[col] = df[col].fillna(0.)
        df[col] = df[col].astype(np.float32)
        df[col] = df[col].apply(lambda x: np.log(x + 1) if x > -1 else -1)
    encoder = LabelEncoder()
    sparse_values_size = []
    for col in sparse_cols:
        df[col] = df[col].fillna("-1")
        df[col] = encoder.fit_transform(df[col])
        sparse_values_size.append(df[col].nunique())
    return df, sparse_values_size


filepath = r"C:\Users\lejianjun\PycharmProjects\deepplus\data\criteo_sample.txt"
# 数据准备
df, integer_cols, categorical_cols = data_preparing(filepath, n_samples=500000)
# 数据处理
df, sparse_values_size = data_process(df, integer_cols, categorical_cols)

print(df.head())
print(sparse_values_size)