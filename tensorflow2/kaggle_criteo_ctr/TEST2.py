import tensorflow as tf
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

data=pd.read_csv(r"C:\Users\lejianjun\git\deepplus\data\criteo\train.txt", sep=" " , header=None)
y=data[0]
X=data.drop([0,40],axis=1)
train_x, test_x, train_y, test_y = train_test_split( X.values, y.values, test_size=0.25, random_state=0)
batch_size = 50

model = tf.saved_model.load(r"C:\Users\lejianjun\git\deepplus\data\criteo\out_model\ctr_net_model")
test_pred=[[test_x.take(range(13),axis=1)[0]],[test_x.take(range(13,39),axis=1)[0]]]
print(test_pred)
y_pred=model(test_pred)
print(y_pred)