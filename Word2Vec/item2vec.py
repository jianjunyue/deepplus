## 第一阶段，先做movies与user的分级分布
import warnings

from tensorboard.notebook import display

warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

import gc
import random
import datetime

import matplotlib.pyplot as plt
path="/Users/apple/PycharmProjects/deepplus/data/ml-20m/"
##数据导入，只需要用到movies电影数据集和评分数据集

# df_movies = pd.read_csv(path+'movies.dat')
# df_ratings = pd.read_csv(path+'/data/ml-20m/ratings.dat')

# 用户信息
# unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
# users = pd.read_table(path+'users.dat', sep='::', header=None, names=unames, engine='python')

# 电影排名
rnames = ['userId', 'movieId', 'rating', 'timestamp']
df_ratings = pd.read_table(path+'ratings.dat', sep='::', header=None, names=rnames,engine='python')

# 电影信息
mnames = ['movieId', 'title', 'genres']
df_movies = pd.read_table(path+'movies.dat', sep='::', header=None, names=mnames, engine='python')

print(df_movies.head())

#将电影数据分别转换为id=>name，name=>id的映射字典
movieId_to_name = pd.Series(df_movies.title.values, index = df_movies.movieId.values).to_dict()
name_to_movieId = pd.Series(df_movies.movieId.values, index = df_movies.title.values).to_dict()

#随机打印5条记录在数据集里
for df in list((df_movies, df_ratings)):
    rand_idx = np.random.choice(len(df), 5, replace=False)
    # display(df.iloc[rand_idx, :])
    print(f"Displaying 5 of the total {str(len(df))} data points")

##分析数据阶段
plt.figure(figsize=(6, 4))
ax = plt.subplot(111)

ax.set_title("Distribution of Movie Ratings")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
plt.xlabel("Movie Rating")
plt.ylabel("Count")
plt.hist(df_ratings['rating'])
plt.show()
##数据集切分
start = datetime.datetime.now()

df_ratings_train, df_ratings_test = train_test_split(df_ratings,
                                                    stratify=df_ratings['userId'], #按userID分布做划分
                                                    random_state=1234,
                                                    test_size=0.30)
print(f"Number of rating data: {str(len(df_ratings_train))}")
print(f"Number of test data: {str(len(df_ratings_test))}")

print(f"Time passed: {str(datetime.datetime.now() - start)}")

##构建item序列，并根据分数来做类别划分，然后根据分组来确定最终的item序列
def rating_splitter(df):
    df['liked'] = np.where(df['rating'] >= 4, 1, 0)  # 输出0/1划分
    df['movieId'] = df['movieId'].astype('str')
    gp_user_like = df.groupby(['liked', 'userId'])

    return ([gp_user_like.get_group(gp)['movieId'].tolist() for gp in gp_user_like.groups])
start = datetime.datetime.now()

pd.options.mode.chained_assignment = None
splitted_movies = rating_splitter(df_ratings_train)

print(f"Time passed: {str(datetime.datetime.now() - start)}")

##使用gensim的word2vec训练item2vec模型
for movie_list in splitted_movies:
    random.shuffle(movie_list)

start = datetime.datetime.now()

model = Word2Vec(sentences=splitted_movies,  # 迭代序列
                 iter=5,  # 迭代次数
                 min_count=4,  # 忽略词频，小于10的将被忽略掉
                 size=32,  # 训练后的向量维度
                 workers=2,  # 设置的线程数
                 sg=1,  # 训练模型的模型选择，1=skip-gram，0=CBOW
                 hs=0,  # 训练代价函数的选择
                 negative=5,  # 负采样
                 window=5)  # 当前词和预测词的最大间隔

print(f"Time passed: {str(datetime.datetime.now() - start)}")
##保存模型，保存了所有模型相关的信息，隐藏权重，词汇频率和模型二叉树，保存为word2vec文本格式，不能追加训练
model.wv.save_word2vec_format(path+"model/item2vec_model_0315.bin", binary=True)
model.wv.save_word2vec_format(path+"model/item2vec_model_0315.txt", binary=False)
##保存模型,可以追加训练
model.save(path+"model/item2vec_model_0315.model")

##模型加载
# gensim.models.KeyedVectors.load_word2vec_format('XX.txt', binary=False)
# gensim.models.KeyedVectors.load_word2vec_format('XX.bin', binary=True)

##追加训练，只能使用model.save的文件,更新模型
# model_add = gensim.models.Word2Vec.load("XXX").train(more_sentences)

##模型评估
# model.accuracy(path+"model/accuracy_item2vec.txt")
def produce_list_of_movieId(list_of_movieName):
    list_of_movie_id = []
    for movieName in list_of_movieName:
        if movieName in name_to_movieId.keys():
            print(f"model:produce_list_of_movieId, movieName: " + movieName)
            list_of_movie_id.append(str(name_to_movieId[movieName]))
    print(f"model:produce_list_of_movieId, list_of_movie_id:{list_of_movie_id}")
    return list_of_movie_id

def recommender(positive_list=None, negative_list=None, topn=20):
    recommend_movie_ls = []
    try:
        if positive_list:
            print(f"model:recommender, positive_list:{positive_list}")
            positive_list = produce_list_of_movieId(positive_list)
        if negative_list:
            print(f"model:recommender, negative_list:{negative_list}")
            negative_list = produce_list_of_movieId(negative_list)

        print(f"model:recommender, positive_list:{positive_list}")
        for movieId, prob in model.wv.most_similar_cosmul(positive=positive_list, negative=negative_list, topn=topn):
            recommend_movie_ls.append(movieId)
    except:
        print("Error")
    return recommend_movie_ls

m_now="Toy Story (1995)"
ls = recommender(positive_list=[m_now], topn=5)
print(f'Recommendation Result based on "{m_now}":')
# if ls:
#     display(df_movies[df_movies['movieId'].isin(ls)])

m_now="Transformers (2007)"
ls = recommender(positive_list=[m_now], topn=5)
print(f'Recommendation Result based on "{m_now}":')
# if ls:
#     display(df_movies[df_movies['movieId'].isin(ls)])

m_now="2012 (2009)"
ls = recommender(positive_list=[m_now], topn=5)
print(f'Recommendation Result based on "{m_now}":')
# display(df_movies[df_movies['movieId'].isin(ls)])

m_now="Titanic (1997)"
ls = recommender(positive_list=[m_now], topn=5)
print(f'Recommendation Result based on "{m_now}":')
# display(df_movies[df_movies['movieId'].isin(ls)])

m_now="Enemy at the Gates (2001)"
ls = recommender(positive_list=[m_now], topn=5)
print(f'Recommendation Result based on "{m_now}":')
# display(df_movies[df_movies['movieId'].isin(ls)])

def print_similarity(name_to_movieId1,name_to_movieId2):
    try:
        similarity=model.wv.similarity(str(name_to_movieId[name_to_movieId1]), str(name_to_movieId[name_to_movieId2]))
        print(name_to_movieId1+" "+name_to_movieId2+" "+similarity)
    except:
        print("Error")

###查看两个向量的相似度
print_similarity('Rain Man (1988)','Truman Show, The (1998)')
print_similarity('Rain Man (1988)','Up (2009)')
print_similarity('Up (2009)','WALL·E (2008)')
print_similarity('2012 (2009)','Enemy at the Gates (2001)')
# print(model.wv.similarity(str(name_to_movieId['Rain Man (1988)']) , str(name_to_movieId['Truman Show, The (1998)'])))
# print(model.wv.similarity(str(name_to_movieId['Rain Man (1988)']) , str(name_to_movieId['Up (2009)'])))
# print(model.wv.similarity(str(name_to_movieId['Up (2009)']) , str(name_to_movieId['WALL·E (2008)'])))
# print(model.wv.similarity(str(name_to_movieId['2012 (2009)']) , str(name_to_movieId['Enemy at the Gates (2001)'])))

#找出不同类的item
def test():
    try:
        diff = model.wv.doesnt_match([str(name_to_movieId['Up (2009)']) ,
                                                  str(name_to_movieId['WALL·E (2008)']),
                                                  str(name_to_movieId['Despicable Me (2010)']),
                                                  str(name_to_movieId['Rain Man (1988)'])])
        df_movies[df_movies['movieId'].astype('str') == diff]

        ##获取相关的向量
        item_vector = model.wv['1234']
        print(item_vector)
        print(df_ratings_test.head())
        print(splitted_movies[0])

        df_test =df_ratings_test.iloc[:10000, :].groupby(['movieId', 'userId'])
        for gp_t in df_test.groups:
            print(gp_t)
            print(df_test.get_group(gp_t)['userId'].tolist())
    except:
        print("Error")
test()
print("end")
