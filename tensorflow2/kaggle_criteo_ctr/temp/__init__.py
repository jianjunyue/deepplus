import os
import sys
# import click
import random
import collections
import pickle
import numpy as np
# import xgboost as lgb
import json
import pandas as pd
from sklearn.metrics import mean_squared_error

# There are 13 integer features and 26 categorical features
# continous_features = range(1, 14)
# categorial_features = range(14, 40)
continous_features=[1,2,3,4,5,6,7,8,9,10,11,12,13]
categorial_features=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
# for i in range(0,14):
    # continous_features[i]=i+1

# for i in range(0,26):
#     categorial_features[i]=i+14


# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]

class ContinuousFeatureGenerator:
    """
    Normalize the integer features to [0, 1] by min-max normalization
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(' ')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = float(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        if self.max[idx] == self.min[idx]:
            return 0.0
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])

class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(' ')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())

            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            temp= self.dicts[i]
            print( self.dicts[i])
            if len(temp)>0:
                vocabs, _a = list(zip(*self.dicts[i]))
                self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
                # self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            # res = self.dicts[idx]['<unk>']
            res =0
            print("")
        else:
            try:
                res = self.dicts[idx][key]
                return res
            except :
                print("")
                return 0
        return 0

    def dicts_sizes(self):
        return list(map(len, self.dicts))


def preprocess(datadir, outdir):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.

    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(os.path.join(datadir, 'train.txt'), continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(
        os.path.join(datadir, 'train.txt'), categorial_features, cutoff=200)  # 200 50

    dict_sizes = dicts.dicts_sizes()
    categorial_feature_offset = [0]
    for i in range(1, len(categorial_features)):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)

    random.seed(0)

    # 90% of the data are used for training, and 10% of the data are used
    # for validation.
    train_ffm = open(os.path.join(outdir, 'train_ffm.txt'), 'w')
    valid_ffm = open(os.path.join(outdir, 'valid_ffm.txt'), 'w')

    train_lgb = open(os.path.join(outdir, 'train_lgb.txt'), 'w')
    valid_lgb = open(os.path.join(outdir, 'valid_lgb.txt'), 'w')

    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(outdir, 'valid.txt'), 'w') as out_valid:
            with open(os.path.join(datadir, 'train.txt'), 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split(' ')
                    continous_feats = []
                    continous_vals = []
                    for i in range(0, len(continous_features)):
                        val = dists.gen(i, features[continous_features[i]])
                        continous_vals.append(
                            "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                        continous_feats.append(
                            "{0:.6f}".format(val).rstrip('0').rstrip('.'))  # ('{0}'.format(val))

                    categorial_vals = []
                    categorial_lgb_vals = []
                    for i in range(0, len(categorial_features)):
                        val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i]
                        categorial_vals.append(str(val))
                        val_lgb = dicts.gen(i, features[categorial_features[i]])
                        categorial_lgb_vals.append(str(val_lgb))

                    continous_vals = ','.join(continous_vals)
                    categorial_vals = ','.join(categorial_vals)
                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write(','.join(
                            [continous_vals, categorial_vals, label]) + '\n')
                        train_ffm.write('\t'.join(label) + '\t')
                        train_ffm.write('\t'.join(
                            ['{}:{}:{}'.format(ii, ii, val) for ii, val in
                             enumerate(continous_vals.split(','))]) + '\t')
                        train_ffm.write('\t'.join(
                            ['{}:{}:1'.format(ii + 13, str(np.int32(val) + 13)) for ii, val in
                             enumerate(categorial_vals.split(','))]) + '\n')

                        train_lgb.write('\t'.join(label) + '\t')
                        train_lgb.write('\t'.join(continous_feats) + '\t')
                        train_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

                    else:
                        out_valid.write(','.join(
                            [continous_vals, categorial_vals, label]) + '\n')
                        valid_ffm.write('\t'.join(label) + '\t')
                        valid_ffm.write('\t'.join(
                            ['{}:{}:{}'.format(ii, ii, val) for ii, val in
                             enumerate(continous_vals.split(','))]) + '\t')
                        valid_ffm.write('\t'.join(
                            ['{}:{}:1'.format(ii + 13, str(np.int32(val) + 13)) for ii, val in
                             enumerate(categorial_vals.split(','))]) + '\n')

                        valid_lgb.write('\t'.join(label) + '\t')
                        valid_lgb.write('\t'.join(continous_feats) + '\t')
                        valid_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

    train_ffm.close()
    valid_ffm.close()

    train_lgb.close()
    valid_lgb.close()

    test_ffm = open(os.path.join(outdir, 'test_ffm.txt'), 'w')
    test_lgb = open(os.path.join(outdir, 'test_lgb.txt'), 'w')

    with open(os.path.join(outdir, 'test.txt'), 'w') as out:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(' ')

                continous_feats = []
                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    continous_vals.append(
                        "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                    continous_feats.append(
                        "{0:.6f}".format(val).rstrip('0').rstrip('.'))  # ('{0}'.format(val))

                categorial_vals = []
                categorial_lgb_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i,
                                    features[categorial_features[i] -
                                             1]) + categorial_feature_offset[i]
                    categorial_vals.append(str(val))

                    val_lgb = dicts.gen(i, features[categorial_features[i] - 1])
                    categorial_lgb_vals.append(str(val_lgb))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)

                out.write(','.join([continous_vals, categorial_vals]) + '\n')

                test_ffm.write('\t'.join(
                    ['{}:{}:{}'.format(ii, ii, val) for ii, val in enumerate(continous_vals.split(','))]) + '\t')
                test_ffm.write('\t'.join(
                    ['{}:{}:1'.format(ii + 13, str(np.int32(val) + 13)) for ii, val in
                     enumerate(categorial_vals.split(','))]) + '\n')

                test_lgb.write('\t'.join(continous_feats) + '\t')
                test_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

    test_ffm.close()
    test_lgb.close()
    return dict_sizes

intput=r"C:\Users\lejianjun\git\deepplus\data\criteo"
outpath=r"C:\Users\lejianjun\git\deepplus\data\criteo\out"

dict_sizes = preprocess(intput,outpath)