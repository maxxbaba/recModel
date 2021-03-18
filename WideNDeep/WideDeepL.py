import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, layers

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from collections import namedtuple, OrderedDict

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embeddings_initializer',
                             'embedding_name',
                             'group_name', 'trainable'])):
    __slots__ = ('name','vocabulary_size','embedding_dim', 'group_name', 'trainable') # 初始化省内存

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embeddings_initializer=None,
                embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype', 'transform_fn'])):
    """ Dense feature
    Args:
        name: feature name,
        dimension: dimension of the feature, default = 1.
        dtype: dtype of the feature, default="float32".
        transform_fn: If not `None` , a function that can be used to transform
        values of the feature.  the function takes the input Tensor as its
        argument, and returns the output Tensor.
        (e.g. lambda x: (x - 3.0) / 4.2).
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", transform_fn=None):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()


class DNN(layers.Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0):
        super(DNN, self).__init__()
        self.dnn_net = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        x = inputs
        for layer in self.dnn_net:
            x = layer(x)
        x = self.dropout(x)
        return x


def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features


def get_linear_logit(features, linear_feature_columns, embed_matrix):
    """

    :param features: dict of inputs {feat:Input()}
    :param linear_feature_columns: namedtuple list of linear part, may consist of both sparse feat and dense feat
    :return: output of linear part -> Linear([sparse_embed, dense_feat_input])
    """
    # extract dense features inputs
    dense_feat_list = []
    for i in range(len(linear_feature_columns)):
        fc = linear_feature_columns[i]
        if isinstance(fc, DenseFeat):
            dense_feat_list.append(features[fc.name])

    sparse_embed_list = []
    for i in range(len(linear_feature_columns)):
        fc = linear_feature_columns[i]
        if isinstance(fc, SparseFeat):
            # getting features own embedding from the embedding lookup dict
            sparse_embed = embed_matrix[fc.name](features[fc.name])
            sparse_embed_list.append(sparse_embed)
    linear_logit = tf.concat([dense_feat_list, sparse_embed_list])
    linear_logit_out = Dense(1)(linear_logit)

    return linear_logit_out

def get_embed_from_matrix(features, fc):
    """
    Look thru embed matrix and return matrix list
    :param features: feat: Input
    :param fc:
    :return:
    """
    return []

def input_from_feature_columns(features, dnn_feature_columns):
    pass

def wdl(linear_feature_columns, dnn_feature_columns, dnn_hidden_units, dnn_activation='relu', task='binary'):
    """

    :param linear_feature_columns:
    :param dnn_feature_columns:
    :param dnn_hidden_units:
    :param dnn_activation:
    :param task:
    :return:
    """
    # 将linear和dnn两侧的特征input综合起来 namedtuple_list -> Input_dict {feat: Input}
    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs = list(features.values())
    linear_logit = get_linear_logit(features, linear_feature_columns) # 得到线性wide部分的输出

    # 组建dnn侧
    sparse_feat_embed_list, dense_feat_list = input_from_feature_columns(features, dnn_feature_columns)

    dnn_input = tf.concat([sparse_feat_embed_list, dense_feat_list])
    dnn_out = DNN(dnn_hidden_units, dnn_activation)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(1, dnn_activation)(dnn_out)

    final_logit = tf.keras.layers.Add()([linear_logit, dnn_logit])

    output = tf.keras.layers.Dense(1, activation='sigmoid')(final_logit)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)

    return model
