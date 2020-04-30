import pandas as pd
from deepctrplus.inputs import SparseFeat, VarLenSparseFeat
import random
import numpy as np
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Model

from deepmatchplus.models import *

def gen_data_set(data, negsample=0):

    # data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, histDF in tqdm(data.groupby('user_id')):
        print(type(histDF))
        print(histDF.head())
        pos_list = histDF['movie_id'].tolist()
        rating_list = histDF['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)
        for i in range(0, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1,len(hist[::-1]),rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist[::-1]),rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]),len(test_set[0]))
    print(len(train_set),len(test_set))

    return train_set,test_set

def gen_model_input(train_set,user_profile,seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', value=0)
    print("----------------------train_seq------------------------")
    print(train_seq)
    print("----------------------train_seq_pad------------------------")
    print(train_seq_pad)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in [  "age" ]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input,train_label

if __name__ == "__main__":
    data = pd.read_csv("../data/movielens_sample1.txt")
    print(data.head())
    # data=data[["user_id", "movie_id", "rating","age" ]].head(10)
    # data.to_csv("../data/movielens_sample1.txt",index=None)
    sparse_features = ["movie_id", "user_id", "age"  ]
    SEQ_LEN = 50
    negsample = 0

    def lbe_print(feature,lbe):
        feature_dict = {}
        for i in range(lbe.classes_.size):
            feature_dict[lbe.classes_[i]]=i+1
        print(feature+":")
        print(feature_dict)
    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    features = ['user_id', 'movie_id', 'age' ]
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        lbe_print(feature, lbe)

    user_profile = data[["user_id",  "age" ]].drop_duplicates('user_id')
    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    print(data.head())
    train_set, test_set = gen_data_set(data, negsample)
    print("----------------------train_set------------------------")
    print(train_set)
    print("----------------------test_set------------------------")
    print(test_set)
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    embedding_dim = 16

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            SparseFeat("age", feature_max_idx['age'], embedding_dim),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    # 3.Define Model and train

    # model = DSSM(user_feature_columns, item_feature_columns)
    model = FM(user_feature_columns,item_feature_columns)

    model.compile(optimizer='adagrad', loss="binary_crossentropy")

    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=256, epochs=1, verbose=1, validation_split=0.0, )

    # 4. Generate user features for testing and full item features for retrieval
    test_user_model_input = test_model_input
    all_item_model_input = {"movie_id": item_profile['movie_id'].values,}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(item_embs.shape)
    print("-------------------------------")
    print(user_embs[0])
    print(item_embs[0])