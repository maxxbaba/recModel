# DeepFM

**1、介绍**

对于CTR问题，被证明的最有效的提升任务表现的策略是特征组合(Feature Interaction), 在CTR问题的探究历史上来看就是如何更好地学习特征组合，进而更加精确地描述数据的特点。可以说这是基础推荐模型到深度学习推荐模型遵循的一个主要的思想。而组合特征大牛们研究过组合二阶特征，三阶甚至更高阶，但是面临一个问题就是随着阶数的提升，复杂度就成几何倍的升高。这样即使模型的表现更好了，但是推荐系统在实时性的要求也不能满足了。所以很多模型的出现都是为了解决另外一个更加深入的问题：如何更高效的学习特征组合？

为了解决上述问题，出现了FM和FFM来优化LR的特征组合较差这一个问题。并且在这个时候科学家们已经发现了DNN在特征组合方面的优势，所以又出现了FNN和PNN等使用深度网络的模型。但是DNN也存在局限性（**网络参数量十分巨大**），所以通过embedidng的预处理将**高维稀疏向量**转化为**低维稠密向量**来降低参数量。

但是还是缺少低阶特征的组合，所以增加了FM模块来补足这一短板。

****

**2、模型结构**



<img src="/Users/jiahongxie/Desktop/GitHub/recModel/DeepFM/pic/1616332941421.jpg" alt="1616332941421" style="zoom:33%;" />

通过这个模型结构图可以看出模型分为几步骤：

**1）**将特征进行embedding映射

**2）**将经过embedding层的特征vector进入FM与MLP部分

**3）**将FM和MLP部分的输出进行加和，经过sigmoid得到最终的输出

****

**3、代码实现**

整体模块：

```python
def DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
```



代码分块：

首先是利用传入的namedtuple信息经过**build_input_features**得到输入的Input()列表，我们建立的是一个有序的字典，这样保证我们取字典values的时候是有序的

```python
def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features
```

**Linear part**的代码：

这里输入的feature_columns是在最开始传入的linear_feature_columns的namedtuple元素。在之后会按照是否是稀疏的特征及一些多值稀疏特征分别处理，最后得到一些经过embedding层的输入。**注意**：这里的稀疏特征是与mlp部分的分开进行embedding的，因为在这里面的稀疏特征会进行一个embed_dim的维度替换，转换成维度1。以下部分看起来很好理解，**但是自己操作的时候要注意特征维度的变化拼接，虽然tf.concat可以sparse_feat与linear_feat各自拼接起来，但是这两个类别一起拼接的时候，容易出现维度错误的信息。所以需要进行一个Flatten()的操作。**

```python
def get_linear_logit(features, feature_columns, units=1, use_bias=False, seed=1024, prefix='linear',
                     l2_reg=0):
    linear_feature_columns = copy(feature_columns)
    for i in range(len(linear_feature_columns)):
        if isinstance(linear_feature_columns[i], SparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(embedding_dim=1,
                                                                           embeddings_initializer=Zeros())
        if isinstance(linear_feature_columns[i], VarLenSparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(
                sparsefeat=linear_feature_columns[i].sparsefeat._replace(embedding_dim=1,
                                                                         embeddings_initializer=Zeros()))

    linear_emb_list = [input_from_feature_columns(features, linear_feature_columns, l2_reg, seed,
                                                  prefix=prefix + str(i))[0] for i in range(units)]
    _, dense_input_list = input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix)

    linear_logit_list = []
    for i in range(units):

        if len(linear_emb_list[i]) > 0 and len(dense_input_list) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias, seed=seed)([sparse_input, dense_input])
        elif len(linear_emb_list[i]) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias, seed=seed)(sparse_input)
        elif len(dense_input_list) > 0:
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias, seed=seed)(dense_input)
        else:
            # raise NotImplementedError
            return add_func([])
        linear_logit_list.append(linear_logit)

    return concat_func(linear_logit_list)
```

FM部分的操作:

```python
class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)

```

