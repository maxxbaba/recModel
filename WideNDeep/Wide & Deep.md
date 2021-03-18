# *Wide & Deep**

## **1、简述**

线性模型利用手动构造特征组合的方式来使模型具有“记忆性”，即对构造的特征组合具有比较高的敏感度。但是缺点也是因为这个特点比较明显：  

​		1）特征工程需要比较有经验的业务人士参与构造，才能达到比较好的效果

​		2）模型对于没有出现的特征组合的泛化度不够

再此基础上，研究者通过引入深度神经网络来提升模型的泛化能力，利用神经元特征交叉的方式让模型学到一些隐藏在特征背后的组合模式。由此产生了 Wide&Deep模型，组合线性模型和DNN的特点使模型同时具有“记忆性”和“泛化性”。

## **2、模型结构和原理**

<div align=center>
<img src="http://ryluo.oss-cn-chengdu.aliyuncs.com/Javaimage-20200910214310877.png" alt="image-20200910214310877" style="zoom:65%;" />
</div>



- wide部分是一个广义的线性模型，输入的特征主要有两部分组成，一部分是原始的部分特征，另一部分是原始特征的交叉特征(cross-product transformation)，对于交互特征可以定义为：
  $$
  \phi_{k}(x)=\prod_{i=1}^d x_i^{c_{ki}}, c_{ki}\in \{0,1\}
  $$
  $c_{ki}$是一个布尔变量，当第i个特征属于第k个特征组合时，$c_{ki}$的值为1，否则为0，$x_i$是第i个特征的值，大体意思就是两个特征都同时为1这个新的特征才能为1，否则就是0，说白了就是一个特征组合。用原论文的例子举例：

  > AND(user_installed_app=QQ, impression_app=WeChat)，当特征user_installed_app=QQ,和特征impression_app=WeChat取值都为1的时候，组合特征AND(user_installed_app=QQ, impression_app=WeChat)的取值才为1，否则为0。

  对于wide部分训练时候使用的优化器是带$L_1$正则的FTRL算法(Follow-the-regularized-leader)，而L1 FTLR是非常注重模型稀疏性质的，也就是说W&D模型采用L1 FTRL是想让Wide部分变得更加的稀疏，即Wide部分的大部分参数都为0，这就大大压缩了模型权重及特征向量的维度。**Wide部分模型训练完之后留下来的特征都是非常重要的，那么模型的“记忆能力”就可以理解为发现"直接的"，“暴力的”，“显然的”关联规则的能力。**例如Google W&D期望wide部分发现这样的规则：**用户安装了应用A，此时曝光应用B，用户安装应用B的概率大。**

- Deep部分是一个DNN模型，输入的特征主要分为两大类，一类是数值特征(可直接输入DNN)，一类是类别特征(需要经过Embedding之后才能输入到DNN中)，Deep部分的数学形式如下：
  $$
  a^{(l+1)} = f(W^{l}a^{(l)} + b^{l})
  $$
  **我们知道DNN模型随着层数的增加，中间的特征就越抽象，也就提高了模型的泛化能力。**对于Deep部分的DNN模型作者使用了深度学习常用的优化器AdaGrad，这也是为了使得模型可以得到更精确的解。

## 3、代码设计

Wide侧记住的是历史数据中那些**常见、高频**的模式，是推荐系统中的“**红海**”。实际上，Wide侧没有发现新的模式，只是学习到这些模式之间的权重，做一些模式的筛选。正因为Wide侧不能发现新模式，因此我们需要**根据人工经验、业务背景，将我们认为有价值的、显而易见的特征及特征组合，喂入Wide侧**

Deep侧就是DNN，通过embedding的方式将categorical/id特征映射成稠密向量，让DNN学习到这些特征之间的**深层交叉**，以增强扩展能力。

**1）整理特征**

根据特征理解，需要将特征分为连续值特征DenseFeat和稀疏特征SparseFeat。以下代码可以定义这两类，这两个类都是继承namedtuple，这里这么处理可以很清楚的把一个特征的类型，处理方式等一系列信息放入了namedtuple里。在后续特征处理的时候，可以很清晰的根据这个类里的方式对特定的特征进行不一样的处理。 整体代码中输入的**linear_feature_columns, dnn_feature_columns**都是由这两个类组成的list。

```python
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embeddings_initializer', 'embedding_name', 'group_name', 'trainable'])):
    __slots__ = () # 初始化省内存，并限定此类只能有这几个key

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
```



**2）embedding lookup设计**

**create_embedding_dict**和**create_embedding_matrix**对所有稀疏的特征都初始化一个embedding，并利用字典来存储。这里样子处理可以方便后续根据特征名字找到对应的embedding向量。这里把一个事情分成两个function来处理是为了代码解耦，后期维护更新不会出现牵一发而动全身的情况。

```python
def create_embedding_dict(sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding # 返回的是一个{feat: Embedding}字典
  
  def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    from . import feature_column as fc_lib

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict
```

**embedding_lookup**输出是一个按照group分组的字典，字典的value保存的是当前组别的所有sparsefeat进行embedding之后的结果

```python
def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list) # 字典是以一个的value是list形态的
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name] # 这个是Input()

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx)) # append 保证了有序
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict # 返回的是一个dict {group: [embedding, embedding]}
```



**3）构造网络输入层input_from_feature_columns**

可以看到这个function返回的是两个list，一个是稀疏特征SparseFeat的Input()经过embedding之后的list，一个DensefFeat构建的Input() list。得到这两个list可以concat在一起作为神经网络层的输入

```python
def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
   
    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                    seq_mask_zero=seq_mask_zero)
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list

```



**4）模型代码**

```python
def WDL(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',
        task='binary'):
    """Instantiates the Wide&Deep Learning architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
		# 得到feature input的字典 {feat:Input()}
    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values()) # 将input()作为模型的inputs

    # 线性part得到一个逻辑输出
    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)
		
    # sparse_embedding_list, dense_value_list分别是稀疏特征的input()经过embedding之后的list和连续值特征组成的list
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)
    # 深度网络层得到一个输出
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)

    final_logit = add_func([dnn_logit, linear_logit])

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
```

