import tensorflow as tf
from collections import namedtuple
from tensorflow.keras.layers import Input, Embedding, Flatten, Add, Dense, Layer, Activation, Dropout
from tensorflow.keras.models import Model

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])

def build_input(feature_columns):
    sparse_feat = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    dense_feat = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []

    features = {}
    for fc in sparse_feat:
        input = Input(shape=(1,), name=fc.name)
        features[fc.name] = input

    for fc in dense_feat:
        input = Input(shape=(fc.dimension), name=fc.name)
        features[fc.name] = input

    return features


def get_bi_output(sparse_feature_columns, features):

    # make sparse_emb_list
    sparse_embed_list = []
    # print(sparse_feature_columns)
    for fc in sparse_feature_columns:
        emb = Embedding(input_dim=fc.voc_size, output_dim=fc.embedding_dim, name='sparse_embed_'+fc.name)
        sparse_embed_list.append(emb(features[fc.name])) # [(None, 1, embed_dim), (None, 1, embed_dim)]
    sparse_embed_list = tf.concat(sparse_embed_list, axis=1) # B x n x k

    # bi = fm 0.5 * (sqaure_of_sum - sum_of_square)
    sqaure_of_sum = tf.square(tf.reduce_sum(sparse_embed_list, axis=1, keepdims=True))
    sum_of_square = tf.reduce_sum(tf.square(sparse_embed_list), axis=1, keepdims=True)
    bi_output = 0.5 * (sqaure_of_sum - sum_of_square)
    return bi_output


def nfm(dense_feature_columns, sparse_feature_columns, dropout=0, hidden_units=[64, 32]):
    sparse_features = build_input(sparse_feature_columns)
    dense_features = build_input(dense_feature_columns)
    inputs = list(sparse_features.values()) + list(dense_features.values())

    # bi_output 是一个densefeat和sparsefet拼接向量进过简化版fm层后的输出向量
    # dim= B x (sparse_len*dim + dense_len)
    bi_output = get_bi_output(sparse_feature_columns, sparse_features)

    bi_output = Flatten()(bi_output)
    # print(bi_output)
    dense_value_list = tf.concat(list(dense_features.values()), axis=1)  # dense_value_list is dict
    # dense_value_list = Flatten()(dense_value_list)
    dnn_input = tf.concat([bi_output, dense_value_list], axis=1)
    # dnn_out = DNN(dropout=dropout, hidden_units=hidden_units)(dnn_input)

    # dnn layer
    for unit in hidden_units:
        dnn_input = Dense(units=unit, activation='relu')(dnn_input)
        dnn_input = Dropout(dropout)(dnn_input)

    outputs = Dense(1, activation='sigmoid')(dnn_input)

    # outputs = Activation('sigmoid')(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# input --> embedding --> bi --> dnn --> sigmoid

# 简单处理特征，包括填充缺失值，数值处理，类别编码
def data_process(data_df, dense_features, sparse_features):  # dense_features\sparse_features 都是名字
    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)  # 对dense数据进行拉平处理

    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    for f in sparse_features:
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])

    return data_df[dense_features + sparse_features]  # 返回处理好的DataFrame

if __name__ == "__main__":
    # 读取数据
    file = './criteo_sampled_data_1w.csv'
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']

    data = pd.read_csv(file, sep=',', header=None, names=names)

    # 划分dense和sparse特征
    columns = data.columns.values
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if 'C' in feat]

    # 简单的数据预处理
    train_data = data_process(data, dense_features, sparse_features)
    train_data['label'] = data['label']

    # 将特征分组，分成linear部分和dnn部分(根据实际场景进行选择)，并将分组之后的特征做标记（使用DenseFeat, SparseFeat）
    dense_feature_columns = [DenseFeat(feat, 1)for feat in dense_features]

    sparse_feature_columns = [SparseFeat(feat, voc_size=data[feat].nunique(), embedding_dim=4) for i, feat in enumerate(sparse_features)]

    # 构建NFM模型 dense sparse
    dropout = 0.5
    hidden_units = [13, 64]
    history = nfm(dense_feature_columns, sparse_feature_columns, dropout, hidden_units)
    # history.summary()
    history.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

    # 将输入数据转化成字典的形式输入
    train_model_input = {name: data[name] for name in dense_features + sparse_features}
    # 模型训练
    history.fit(train_model_input, train_data['label'].values,
                batch_size=64, epochs=20, validation_split=0.2, )


# keras 的输入是字典还是dataframe如何决定的
# 根据 input的形式来决定的，这里是input = list(dict.values()) 所以输入是字典