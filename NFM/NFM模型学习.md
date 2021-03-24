# NFM模型学习

## **1. 简述**

NFM(Neural Factorization Machines)是2017年由新加坡国立大学的何向南教授等人在SIGIR会议上提出的一个模型，传统的FM模型仅局限于线性表达和二阶交互， 无法胜任生活中各种具有复杂结构和规律性的真实数据， 针对FM的这点不足， 作者提出了一种将FM融合进DNN的策略，通过引进了一个特征交叉池化层的结构，使得FM与DNN进行了完美衔接，这样就组合了FM的建模低阶特征交互能力和DNN学习高阶特征交互和非线性的能力，形成了深度学习时代的神经FM模型(NFM)。

那么NFM具体是怎么做的呢？ 首先看一下NFM的公式：
$$
\hat{y}_{N F M}(\mathbf{x})=w_{0}+\sum_{i=1}^{n} w_{i} x_{i}+f(\mathbf{x})
$$


<img src="/Users/jiahongxie/Desktop/GitHub/recModel/NFM/pic/1616578701034.jpg" alt="1616578701034" style="zoom:75%;" />

## **2.构造原理**

**1）Embedding layer**

​	模型的初衷是加强了二阶特征的交互而提升整体效果（通过特征交叉池化层将经过embedding之后的离散特征送入DNN层），所以需要在模型输入的阶段将高维稀疏特征转化为低维稠密向量。

```python
for fc in sparse_feature_columns:
  emb = Embedding(input_dim=fc.voc_size, output_dim=fc.embedding_dim, name='sparse_embed_'+fc.name)
  sparse_embed_list.append(emb(features[fc.name])) # [(None, 1, embed_dim), (None, 1, embed_dim)]
sparse_embed_list = tf.concat(sparse_embed_list, axis=1) # B x n x k
```

**2）bi layer 交叉池化层**

​	这是一个简化版的FM，与普通的FM相比的区别是在最后没有吧所有维度融合成1，而是保持着embedding的维度。**在实现模型过程中，我没有使用传统的稀疏特征放入DNN，其他放入线性部分，因为这样我感觉与wide&deep模型区别不大。所以我直接在bi层结束的恶时候将dense的向量直接与bi层拼接，一起送入DNN。**

```python
# bi层输出
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
```

```python

bi_output = get_bi_output(sparse_feature_columns, sparse_features)

bi_output = Flatten()(bi_output)
dense_value_list = tf.concat(list(dense_features.values()), axis=1)  # dense_value_list is dict
# 拼接bi层和dense特征的值
dnn_input = tf.concat([bi_output, dense_value_list], axis=1)
```

**3）DNN层得到最后的输出值**

```python
for unit in hidden_units:
  dnn_input = Dense(units=unit, activation='relu')(dnn_input)
  dnn_input = Dropout(dropout)(dnn_input)

  outputs = Dense(1, activation='sigmoid')(dnn_input)
```

## **3. 整体代码**

```python
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
```

