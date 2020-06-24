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
from sklearn.preprocessing import LabelEncoder
from temp.deepfm.deepfm import OneOrder, EmbeddingLayer, TwoOrder, HighOrder, LR
from tqdm import tqdm

def data_preparing(filepath, n_samples=50000):
    df = pd.read_csv(filepath)
    sparse_cols = ['C' + str(i) for i in range(14, 27)]  # 类型
    dense_cols = ['I' + str(i) for i in range(1, 14)]  # 值
    return df, dense_cols, sparse_cols

def data_process(df, integer_cols, categorical_cols):
    for col in integer_cols:
        df[col] = df[col].fillna(0.)
        df[col] = df[col].astype(np.float32)
        df[col] = df[col].apply(lambda x: np.log(x+1) if x > -1 else -1)
    encoder = LabelEncoder()
    sparse_values_size = []
    for col in categorical_cols:
        df[col] = df[col].fillna("-1")
        df[col] = encoder.fit_transform(df[col])
        sparse_values_size.append(df[col].nunique())
    return df, sparse_values_size

def build_model(integer_cols, categorical_cols, sparse_values_size, embedding_dim=10):
    sparse_inputs = [Input(shape=(1,), dtype=tf.int32, name=col) for col in categorical_cols]
    dense_inputs = [Input(shape=(1,), dtype=tf.float32, name=col) for col in integer_cols]
    one_order_outputs = OneOrder(sparse_values_size)([sparse_inputs, dense_inputs])
    embeddings = EmbeddingLayer(sparse_values_size, embedding_dim)([sparse_inputs, dense_inputs])
    two_order_outputs = TwoOrder()(embeddings)
    high_order_outputs = HighOrder(2)(embeddings)
    outputs = LR()([one_order_outputs, two_order_outputs, high_order_outputs])
    model = Model(inputs=[sparse_inputs, dense_inputs], outputs=outputs)
    return model

if __name__ == "__main__":
    filepath = r"C:\Users\lejianjun\PycharmProjects\deepplus\data\criteo_sample.txt"
    # 数据准备
    df, integer_cols, categorical_cols = data_preparing(filepath, n_samples=500000)
    # 数据处理
    df, sparse_values_size = data_process(df, integer_cols, categorical_cols)
    # 生成训练数据
    sparse_inputs = [df[col].values for col in categorical_cols]
    dense_inputs = [df[col].values for col in integer_cols]
    targets = df['label'].astype(np.float32).values
    # 构建模型
    model = build_model(integer_cols, categorical_cols, sparse_values_size)
    model.summary()
    # plot_model(model, "deepfm.png")
    # 训练模型
    es = EarlyStopping(patience=5)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(decay=0.0001),
        metrics=[AUC(), 'accuracy'])
    model.fit(
        sparse_inputs + dense_inputs,
        targets,
        batch_size=256,
        epochs=10,
        validation_split=0.2,
        callbacks=[es])