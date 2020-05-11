import tensorflow as tf
import pandas as pd

# 在线下载汽车效能数据集

dataset_path = tf.keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/autompg.data")
# 利用 pandas 读取数据集，字段有效能（公里数每加仑），气缸数，排量，马力，重量
# 加速度，型号年份，产地
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
 na_values = "?", comment='\t',
 sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
# print(dataset.head())

origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
# 切分为训练集和测试集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
# 移动 MPG 油耗效能这一列为真实标签 Y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

print(train_dataset.shape,train_labels.shape)
print(test_dataset.shape, test_labels.shape)

train_db = tf.data.Dataset.from_tensor_slices((train_dataset.values,train_labels.values)) # 构建 Dataset 对象
train_db = train_db.shuffle(100).batch(32) # 随机打散，批量化

for x,y in train_db:
    print(x)
    print(y)

