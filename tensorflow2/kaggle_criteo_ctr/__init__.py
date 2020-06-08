
from tensorflow2.kaggle_criteo_ctr.CTRNetwork import CTRNetwork
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

data=pd.read_csv(r"C:\Users\lejianjun\git\deepplus\data\criteo\train.txt", sep=" " , header=None)
# print(data.head())
# print(data.describe())
y=data[0]
X=data.drop([0,40],axis=1)
print(type(X))
print(type(y))
train_x, test_x, train_y, test_y = train_test_split( X.values, y.values, test_size=0.25, random_state=0)
train=[train_x.take(range(13),axis=1),train_x.take(range(13,39),axis=1)]
test=[test_x.take(range(13),axis=1),test_x.take(range(13,39),axis=1)]
#
embed_dim = 32 # output_dim
sparse_max = 0 # sparse_feature_dim = 117568 input_dim
sparse_dim = 26 # input_length
dense_dim = 13
out_dim = 400

for i in range(13,39):
    sparse_max+=int(max(X.values.take(i, 1)))+ 1
print(sparse_max)


# optimizer = tf.compat.v1.train.FtrlOptimizer(0.01)  # tf.keras.optimizers.Adam(0.01)
optimizer=tf.keras.optimizers.Ftrl()
ctr_net=CTRNetwork(sparse_max,embed_dim,sparse_dim,out_dim)
ctr_net.compile(loss="mean_squared_error",optimizer=optimizer)
history = ctr_net.fit(x=train,y=train_y, validation_data =(test, test_y),  epochs = 2 )

test_pred=[[test_x.take(range(13),axis=1)[0]],[test_x.take(range(13,39),axis=1)[0]]]
print(test_pred)
y_pred=ctr_net.predict(test_pred)
print(y_pred)


