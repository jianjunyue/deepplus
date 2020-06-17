
##构建item序列，并根据分数来做类别划分
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from tensorboard.notebook import display

# https://github.com/blogchong/data_and_advertisement/blob/master/code/019_embedding/item2vec_train.ipynb
# https://zhuanlan.zhihu.com/p/140956362

df_movies=pd.read_csv(r"C:\Users\lejianjun\PycharmProjects\deepplus\data\movielens_sample.txt")

print(df_movies.head())
df_ratings_train=df_movies

def rating_splitter(df):
    df['liked'] = np.where(df['rating'] >= 4, 1, 0) #输出0/1划分
    df['movieId'] = df['movie_id'].astype('str')
    gp_user_like = df.groupby(['liked', 'user_id'])

splitted_movies = rating_splitter(df_ratings_train)

# 调用gensim进行模型训练：
model = Word2Vec(sentences=splitted_movies, #迭代序列
                iter=5, #迭代次数
                min_count=4, #忽略词频，小于10的将被忽略掉
                size=32,  #训练后的向量维度
                workers=2,  #设置的线程数
                sg=1,  #训练模型的模型选择，1=skip-gram，0=CBOW
                hs=0,  #训练代价函数的选择
                negative=5,  #负采样
                window=5)  #当前词和预测词的最大间隔


# modelvec = Word2Vec().setMinCount(5).setVectorSize(32).setSeed(42).setNumPartitions(40).setNumIterations(1).fit(doc)
# word = modelvec.getVectors()
# print(word)

movieId_to_name = pd.Series(df_movies.title.values, index=df_movies.movieId.values).to_dict()
name_to_movieId = pd.Series(df_movies.movieId.values, index=df_movies.title.values).to_dict()


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
    if positive_list:
        print(f"model:recommender, positive_list:{positive_list}")
        positive_list = produce_list_of_movieId(positive_list)
    if negative_list:
        print(f"model:recommender, negative_list:{negative_list}")
        negative_list = produce_list_of_movieId(negative_list)

    print(f"model:recommender, positive_list:{positive_list}")
    for movieId, prob in model.wv.most_similar_cosmul(positive=positive_list, negative=negative_list, topn=topn):
        recommend_movie_ls.append(movieId)
    return recommend_movie_ls

m_now="Toy Story 3 (2010)"
ls = recommender(positive_list=[m_now], topn=5)
print(f'Recommendation Result based on "{m_now}":')
display(df_movies[df_movies['movieId'].isin(ls)])