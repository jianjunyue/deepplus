# coding=UTF-8
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
from sklearn.metrics import make_scorer
from xgboost.sklearn import XGBClassifier

# https://github.com/princewen/tensorflow_practice/blob/master/recommendation/GBDT%2BLR-Demo/GBDT_LR.py

path= r"/data/criteo/out/train_lgb.txt"
data=pd.read_csv(path, header=None,sep = '\t')
train=data.drop(0,axis=1)
y=data[0]

# print(train.head())
# print(y)

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=1)

xgb = XGBClassifier(seed=27,objective= 'binary:logistic',colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             nthread = -1)

xgb.fit(X_train,y_train)
fea=xgb.feature_importances_
print(fea)

#sklearn接口生成的新特征
train_new_feature= xgb.apply(X_train)#每个样本在每颗树叶子节点的索引值
test_new_feature= xgb.apply(X_test)
train_new_feature2 = DataFrame(train_new_feature)
test_new_feature2 = DataFrame(test_new_feature)

print("新的特征集(自带接口)：",train_new_feature2.shape)
print(train_new_feature2.head())