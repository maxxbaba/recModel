import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, ReLU, Layer, Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


# 构建模型
class ResidualUnit(Layer):
    def __init__(self, hidden_units, dim_inputs): # 这个层有多少节点，inputs的维度是如何的
        super(ResidualUnit, self).__init__() # __init__()大多部分继承自Layer
        self.dense_1 = Dense(units=hidden_units, activation='relu')
        self.dense_2 = Dense(units=dim_inputs) # 最后一层的dimension要和input一致才能相加
        self.relu = ReLU()
    
    def call(self, inputs):
        x = inputs
        x = self.dense_1(x)
        x = self.dense_2(x)
        outputs = self.relu(x + inputs)
        return outputs      


class DeepCrossing(keras.Model):
    def __init__(self, feature_cols, hidden_units, dropout_rate=0, embed_reg=1e-4): # 需要考虑使用的时候是传入什么参数，比如数据传入形式、神经网络层级、以及一些可调整的参数dropout之类的
        super(DeepCrossing, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_cols # 把传入的col拆开
        self.embedding = {  # embedding字典
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                        input_length=1,
                                        output_dim=feat['embed_dim'],
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_cols)
        }
        
             
        embed_dim = sum([feat['embed_dim'] for feat in self.sparse_feature_cols])
        dim_stack = embed_dim + len(self.dense_feature_cols)
        self.res = [ResidualUnit(unit, dim_stack) for unit in hidden_units]
        self.res_dropout = Dropout(dropout_rate)
        self.dense = Dense(1, activation='sigmoid')
        
    
    def call(self, inputs):
        # input分两个--dense和sparse
        dense_inputs, sparse_inputs = inputs
        
        print('======== show inputs in the call function ========')
        print(inputs)
        print('============ show dense_inputs format ============')
        print(dense_inputs)
        print('============ show sparse_inputs format ===========')
        
        
        
        sparse_inputs_embed = tf.concat([self.embedding['embed_{}'.format(i)](sparse_inputs[:,i]) \
                                         for i in range(sparse_inputs.shape[1])], axis=-1)
        stacking_layer_inputs = tf.concat([sparse_inputs_embed, dense_inputs], axis=-1)
        r = stacking_layer_inputs
        for residual in self.res:
            r = residual(r)
        r = self.res_dropout(r)
#         output = Dense(1, activation='sigmoid')(r)
#         r = Dense(10)(r)
        output = self.dense(r)
#         output = tf.nn.sigmoid(self.dense(r))
        return output  


# 构建数据集合
def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}

file = './criteo_sampled_data_1w.csv'

sample_num = 5000
test_size = 0.2
embed_dim = 10

names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']

data_df = pd.read_csv(file, sep=',', header=None, names=names)

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data_df[sparse_features] = data_df[sparse_features].fillna('-1')
data_df[dense_features] = data_df[dense_features].fillna(0)

for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

mms = MinMaxScaler(feature_range=(0, 1))
data_df[dense_features] = mms.fit_transform(data_df[dense_features])

feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                  [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim)
                    for feat in sparse_features]]

train, test = train_test_split(data_df, test_size=test_size)

train_X = [train[dense_features].values.astype('float32'), train[sparse_features].values.astype('int32')]
train_y = train['label'].values.astype('int32')
test_X = [test[dense_features].values.astype('float32'), test[sparse_features].values.astype('int32')]
test_y = test['label'].values.astype('int32')


# 模型测试
learning_rate=1e-4
hidden_units = [10, 10]

d_c = DeepCrossing(feature_columns, hidden_units)
d_c.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=[AUC()])

epochs=10
batch_size=1024
d_c.fit(train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_auc', patience=100, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1)

d = d_c.predict(test_X)