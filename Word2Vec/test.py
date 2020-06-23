import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import numpy as np
from pandasql import sqldf
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from gensim.models import doc2vec
import random
import datetime
pysqldf = lambda q: sqldf(q, globals())
##数据导入，只需要用到movies电影数据集和评分数据集
# 电影排名

path="/Users/apple/PycharmProjects/deepplus/data/ml-20m/"
rnames = ['userId', 'movieId', 'rating', 'timestamp']
df_ratings = pd.read_table(path+'ratings.dat', sep='::', header=None, names=rnames,engine='python')

df_ratings['liked'] = np.where(df_ratings['rating'] >= 4, 1, 0)  # 输出0/1划分
df_ratings['movieId'] = df_ratings['movieId'].astype('str')
df_ratings['userId_liked'] =df_ratings['userId'].astype('str')+"_"+df_ratings['liked'].astype('str')
print("--------df_movies---------")
print(df_ratings.head())
df_ratings.to_csv(path+"df_ratings_input.csv", index=None)



path="/Users/apple/PycharmProjects/deepplus/data/ml-20m/"
df_ratings=pd.read_csv(path+"df_ratings_input.csv")
splitted_movies=[]
for val in df_ratings.groupby("userId_liked"):
    strid=""
    movie_list= val[1]['movieId'].astype('str').tolist()
    random.shuffle(movie_list)
    for id in movie_list:
        strid+=","+id
    # print(movie_list)
    splitted_movies.append(strid)
print(splitted_movies[0])
name=['movieId_list']
test=pd.DataFrame(columns=name,data=splitted_movies)
test.to_csv(path+"df_ratings_list_input.csv", index=None)
print(test.head())


# model = Word2Vec(sentences=splitted_movies,  # 迭代序列
#                  iter=5,  # 迭代次数
#                  min_count=4,  # 忽略词频，小于10的将被忽略掉
#                  size=32,  # 训练后的向量维度
#                  workers=2,  # 设置的线程数
#                  sg=1,  # 训练模型的模型选择，1=skip-gram，0=CBOW
#                  hs=0,  # 训练代价函数的选择
#                  negative=5,  # 负采样
#                  window=5)  # 当前词和预测词的最大间隔
#
# ##保存模型，保存了所有模型相关的信息，隐藏权重，词汇频率和模型二叉树，保存为word2vec文本格式，不能追加训练
# model.wv.save_word2vec_format(path+"model/item2vec_model_0620.bin", binary=True)
# model.wv.save_word2vec_format(path+"model/item2vec_model_0620.txt", binary=False)

##模型加载
# model=gensim.models.KeyedVectors.load_word2vec_format(path+"model/item2vec_model_0620.txt", binary=False)