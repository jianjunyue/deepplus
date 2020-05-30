import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
from tensorflow2.movie.mv_network_model import mv_network_model

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('/Users/apple/PycharmProjects/deepplus/data/out/preprocess.p', mode='rb'))

#嵌入矩阵的维度
embed_dim = 32
#用户ID个数
uid_max = max(features.take(0,1)) + 1 # 6040
#性别个数
gender_max = max(features.take(2,1)) + 1 # 1 + 1 = 2
#年龄类别个数
age_max = max(features.take(3,1)) + 1 # 6 + 1 = 7
#职业个数
job_max = max(features.take(4,1)) + 1# 20 + 1 = 21
#电影ID个数
movie_id_max = max(features.take(1,1)) + 1 # 3952
#电影类型个数
movie_categories_max = max(genres2int.values()) + 1 # 18 + 1 = 19
#电影名单词个数
movie_title_max = len(title_set) # 5216
#对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"
#电影名长度
sentences_size = title_count # = 15
#文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
#文本卷积核数量
filter_num = 8
#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}

print("uid_max:"+str(uid_max)+" , gender_max:"+str(gender_max)+" , age_max:"+str(age_max)+" , job_max:"+str(job_max)+" , movie_id_max:"+str(movie_id_max)
      +" , movie_categories_max:"+str(movie_categories_max)+" , movie_title_max:"+str(movie_title_max))

# 将数据集分成训练集和测试集，随机种子不固定
train_X, test_X, train_y, test_y = train_test_split(features,targets_values,  test_size=0.2, random_state=0)

mv_net=mv_network_model()
# mv_net=WideDeepModel()

mv_net.compile(loss="mean_squared_error",
              optimizer="adam"
              )

print(train_X.take([0],axis=1))

tf_train=train_X.take([0,1,2,3,4],axis=1).astype('float32')
tf_test= test_X.take([0,1,2,3,4],axis=1).astype('float32')

split0, split1, split2, split3, split4 = train_X.take([0], axis=1).astype('float32'), train_X.take([1], axis=1).astype(
    'float32'), train_X.take([2], axis=1).astype('float32'), train_X.take([3], axis=1).astype('float32'), train_X.take(
    [4], axis=1).astype('float32')

split0_test, split1_test, split2_test, split3_test, split4_test = test_X.take([0], axis=1).astype('float32'), test_X.take([1], axis=1).astype(
    'float32'), test_X.take([2], axis=1).astype('float32'), test_X.take([3], axis=1).astype('float32'), test_X.take(
    [4], axis=1).astype('float32')

# split0, split1, split2, split3, split4 = tf.split(tf_train, [1, 1, 1, 1, 1], 1)

# split0_test, split1_test, split2_test, split3_test, split4_test = tf.split(test_X, [1, 1, 1, 1, 1], 1)

# 未设置频率 默认每个epoch验证一次
# history = mv_net.fit(x=tf_train,y=train_y, validation_data =(tf_test, test_y),  epochs = 2 )
history = mv_net.fit(x=[split0, split1, split2, split3, split4],y=train_y, validation_data =([split0_test, split1_test, split2_test, split3_test, split4_test], test_y),  epochs = 2 )

print("end")
