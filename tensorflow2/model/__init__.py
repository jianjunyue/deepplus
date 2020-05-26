import pandas as pd
import numpy as np

path="C:\\Users\\lejianjun\\git\\deepplus\\data\\"
train_file =path+ 'HanXiaoyang\\data\\train.txt'

train=pd.read_csv(train_file )
print(train.describe())

# data=train.unique()
print(train[0].value_counts())