# DeepCrossing

<img src="/Users/jiahongxie/Desktop/GitHub/recModel/DeepCrossing/DeepCrossing.png" alt="	" style="zoom:50%;" />

## 1、简介

DeepCrossing是微软2016年提出的模型，其应用场景是微软搜索引擎Bing中的搜索广告推荐。Deep Crossing的优化目标是预测用户对某一个广告的点击率。这是一个把深度学习框架应用于推荐系统的端到端模型

## 2、模型结构原理

​		**1）Embedding Layer**

​			将特征送入Embedding层。通常想要将类别型特征送入神经网络学习，都需要先将这类离散特征转化为稀疏的向量表示。但是特征过于稀疏并不利于模型参数的学习，所以利用embedding层将高维稀疏特征转化为低维稠密向量。若是连续值特征，可以直接当做一位的向量在stacking层与低维稠密的向量拼接。

```python
feat_embedding = tf.keras.layers.Embedding(input_dim,
                                           output_dim,
                                           embeddings_initializer="uniform",
                                           embeddings_regularizer=None,
                                           activity_regularizer=None,
                                           embeddings_constraint=None,
                                           mask_zero=False,
                                           input_length=None,
                                           **kwargs)
```


​		**2）Stacking Layer**

​			这一层是将embedding之后的特征拼接起来，形成一个1 x n的向量，作为DNN的输入。通常是使用Concatnate()进行拼接

```python
#将所有的dense特征拼接到一起
dense_dnn_list = list(dense_input_dict.values()) # 这里的dense_input_dict是一个字典，里面是{feat: Input()}, 这里将不同的Input放入list里面[Input(), Input(), ...]

dense_dnn_inputs = Concatenate(axis=1)(dense_dnn_list) # B x n (n表示数值特征的数量)

# 因为需要将其与dense特征拼接到一起所以需要Flatten，不进行Flatten的Embedding层输出的维度为：Bx1xdim
sparse_dnn_list = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=True) 

sparse_dnn_inputs = Concatenate(axis=1)(sparse_dnn_list) # B x n * dim (n表示类别特征的数量，dim表示embedding的维度)

# 将dense特征和Sparse特征拼接到一起
dnn_inputs = Concatenate(axis=1)([dense_dnn_inputs, sparse_dnn_inputs]) # B x (n + m*dim)
```

​		**3）Multiple Residual Units Layer**

<img src="/Users/jiahongxie/Desktop/GitHub/recModel/DeepCrossing/ResidualUnits.png" alt="	" style="zoom: 50%;" />

​			字面意思这是一个有多个残差网络组成的DNN层，Deep Crossing简化了这个残差单元，只是用两层神经网络构建残差网络。			

```python
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
        outputs = self.relu(x + inputs) # 得到的输出要和输入相加，再进入激活函数。这就是残差操作
        return outputs      
```

​		**4）Scoring Layer**

​			这个作为输出层，往往采用sigmoid函数对输出进行处理得到一个概率值的输出，再利用优化算法对目标进行拟合

```python
self.dense = Dense(1, activation='sigmoid')
```

**完整代码见github**

