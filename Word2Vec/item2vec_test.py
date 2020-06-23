import warnings

import gensim
import pandas as pd
import numpy as np

warnings.filterwarnings(action='ignore')
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
path="/Users/apple/PycharmProjects/deepplus/data/ml-20m/"
dt=pd.read_csv(path+"df_ratings_list_input.csv" )
splitted_movies=[]
for ids in dt.values:
    list= ids[0].split(",")[1:]
    splitted_movies.append(list)
# 电影信息
mnames = ['movieId', 'title', 'genres']
df_movies = pd.read_table(path+'movies.dat', sep='::', header=None, names=mnames, engine='python')
#将电影数据分别转换为id=>name，name=>id的映射字典
movieId_to_name = pd.Series(df_movies.title.values, index = df_movies.movieId.values).to_dict()
name_to_movieId = pd.Series(df_movies.movieId.values, index = df_movies.title.values).to_dict()
def movieName_to_movieId(movieName):
    print(f"movieName:{movieName}, movieId:{str(name_to_movieId[movieName])}")
    return str(name_to_movieId[movieName])
def movieId_to_movieName(movieId):
    # print(f"movieId:{movieId}, movieName:{str(movieId_to_name[int(movieId)])}")
    return str(movieId_to_name[int(movieId)])
#splitted_movies：userid下，好评下（或差评下），movieid集
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
model=gensim.models.KeyedVectors.load_word2vec_format(path+"model/item2vec_model_0620.txt", binary=False)
# gensim.models.KeyedVectors.load_word2vec_format('XX.bin', binary=True)
#

def rec(name):
    movieId = movieName_to_movieId(movieName)
    for movieId, prob in model.wv.most_similar_cosmul(positive=movieId, negative=None, topn=5):
        print(f" {movieId}, {movieId_to_movieName(movieId)}, {prob}")

movieName="Toy Story (1995)"
movieName="Star Wars: Episode VI - Return of the Jedi (1983)"
rec(movieName)