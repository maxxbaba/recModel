# DIN

## 1、动机

Deep Interest Network(DIIN)是2018年阿里巴巴提出来的模型， 该模型基于业务的观察，从实际应用的角度进行改进，相比于之前很多“学术风”的深度模型， 该模型更加具有业务气息。该模型的应用场景是阿里巴巴的电商广告推荐业务， 这样的场景下一般**会有大量的用户历史行为信息**， 这个其实是很关键的，因为DIN模型的创新点或者解决的问题就是使用了注意力机制来对用户的兴趣动态模拟， 而这个模拟过程存在的前提就是用户之前有大量的历史行为了，这样我们在预测某个商品广告用户是否点击的时候，就可以参考他之前购买过或者查看过的商品，这样就能猜测出用户的大致兴趣来，这样我们的推荐才能做的更加到位，所以这个模型的使用场景是**非常注重用户的历史行为特征（历史购买过的商品或者类别信息）**，也希望通过这一点，能够和前面的一些深度学习模型对比一下。

在个性化的电商广告推荐业务场景中，也正式由于用户留下了大量的历史交互行为，才更加看出了之前的深度学习模型(作者统称Embeding&MLP模型)的不足之处。如果学习了前面的各种深度学习模型，就会发现Embeding&MLP模型对于这种推荐任务一般有着差不多的固定处理套路，就是大量稀疏特征先经过embedding层， 转成低维稠密的，然后进行拼接，最后喂入到多层神经网络中去。 

这些模型在这种个性化广告点击预测任务中存在的问题就是**无法表达用户广泛的兴趣**，因为这些模型在得到各个特征的embedding之后，就蛮力拼接了，然后就各种交叉等。这时候根本没有考虑之前用户历史行为商品具体是什么，究竟用户历史行为中的哪个会对当前的点击预测带来积极的作用。 而实际上，对于用户点不点击当前的商品广告，很大程度上是依赖于他的历史行为的，王喆老师举了个例子

>假设广告中的商品是键盘， 如果用户历史点击的商品中有化妆品， 包包，衣服， 洗面奶等商品， 那么大概率上该用户可能是对键盘不感兴趣的， 而如果用户历史行为中的商品有鼠标， 电脑，iPad，手机等， 那么大概率该用户对键盘是感兴趣的， 而如果用户历史商品中有鼠标， 化妆品， T-shirt和洗面奶， 鼠标这个商品embedding对预测“键盘”广告的点击率的重要程度应该大于后面的那三个。

这里也就是说如果是之前的那些深度学习模型，是没法很好的去表达出用户这广泛多样的兴趣的，如果想表达的准确些， 那么就得加大隐向量的维度，让每个特征的信息更加丰富， 那这样带来的问题就是计算量上去了，毕竟真实情景尤其是电商广告推荐的场景，特征维度的规模是非常大的。 并且根据上面的例子， 也**并不是用户所有的历史行为特征都会对某个商品广告点击预测起到作用**。所以对于当前某个商品广告的点击预测任务，没必要考虑之前所有的用户历史行为。 

这样， DIN的动机就出来了，在业务的角度，我们应该自适应的去捕捉用户的兴趣变化，这样才能较为准确的实施广告推荐；而放到模型的角度， 我们应该**考虑到用户的历史行为商品与当前商品广告的一个关联性**，如果用户历史商品中很多与当前商品关联，那么说明该商品可能符合用户的品味，就把该广告推荐给他。而一谈到关联性的话， 我们就容易想到“注意力”的思想了， 所以为了更好的从用户的历史行为中学习到与当前商品广告的关联性，学习到用户的兴趣变化， 作者把注意力引入到了模型，设计了一个"local activation unit"结构，利用候选商品和历史问题商品之间的相关性计算出权重，这个就代表了对于当前商品广告的预测，用户历史行为的各个商品的重要程度大小， 而加入了注意力权重的深度学习网络，就是这次的主角DIN， 下面具体来看下该模型。



## 2、原理及构造

![1616805101962](/Users/jiahongxie/Desktop/GitHub/recModel/DIN/pic/1616805101962.jpg)

​		左边是传统的Embedding+MLP模型，右边是DIN模型。可以看出这两模型的唯一区别是在concatenate之前，qurey item的向量会与用户侧的特征向量经过activation unit的计算，得到一个权重，再与用户侧特征向量相乘。通过这个操作可以再concatenate层拼接的时候体现出用户历史点击记录对query item的兴趣。

​		**1）Embedding层**

​				这里和以往的embedding一样。利用keras自带的embedding layer将onehot之后的cate特征转化为向量。

​		**2）Activation Unit**

​				将query item和用户的历史行为记录进行out product的操作，然后得到向量在作用于用户的历史行为记录的向量上。具体实现时，并没有把每一个行为给拆分出来单独进行activation操作，而是将整体历史行为作为一个tensor与query item的向量相乘，得到权重w。代码如下：

```python
class LocalActivationUnit(Layer):

    def __init__(self, hidden_units=(256, 128, 64), activation='prelu'):
        super(LocalActivationUnit, self).__init__()
        self.hidden_units = hidden_units
        self.linear = Dense(1)
        self.dnn = [Dense(unit, activation=PReLU() if activation == 'prelu' else Dice()) for unit in hidden_units]

    def call(self, inputs):
        # query: B x 1 x emb_dim  keys: B x len x emb_dim
        query, keys = inputs

        # 获取序列长度
        keys_len = keys.get_shape()[1]

        queries = tf.tile(query, multiples=[1, keys_len, 1])  # (None, len, emb_dim)

        # 将特征进行拼接
        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # B x len x 4*emb_dim

        # 将原始向量与外积结果拼接后输入到一个dnn中
        att_out = att_input
        for fc in self.dnn:
            att_out = fc(att_out)  # B x len x att_out

        att_out = self.linear(att_out)  # B x len x 1
        att_out = tf.squeeze(att_out, -1)  # B x len

        return att_out


class AttentionPoolingLayer(Layer):
    def __init__(self, att_hidden_units=(256, 128, 64)):
        super(AttentionPoolingLayer, self).__init__()
        self.att_hidden_units = att_hidden_units
        self.local_att = LocalActivationUnit(self.att_hidden_units)

    def call(self, inputs):
        # keys: B x len x emb_dim, queries: B x 1 x emb_dim
        #  for i in range(len(keys_embed_list)):
        # 		seq_emb = AttentionPoolingLayer()([query_embed_list[i], keys_embed_list[i]])
        # 		dnn_seq_input_list.append(seq_emb)
        # 这个keys的维度应该是 B x 1 x emb_dim
        queries, keys = inputs
        print('keys: ', keys.shape)
        print('queries: ', queries.shape)

        # 获取行为序列embedding的mask矩阵，将Embedding矩阵中的非零元素设置成True，
        key_masks = tf.not_equal(keys[:, :, 0], 0)  # B x len
        # key_masks = keys._keras_mask # tf的有些版本不能使用这个属性，2.1是可以的，2.4好像不行

        # 获取行为序列中每个商品对应的注意力权重
        attention_score = self.local_att([queries, keys])  # B x len

        # 去除最后一个维度，方便后续理解与计算
        # outputs = attention_score
        # 创建一个padding的tensor, 目的是为了标记出行为序列embedding中无效的位置
        paddings = tf.zeros_like(attention_score)  # B x len

        # outputs 表示的是padding之后的attention_score
        outputs = tf.where(key_masks, attention_score, paddings)  # B x len

        # 将注意力分数与序列对应位置加权求和，这一步可以在
        outputs = tf.expand_dims(outputs, axis=1)  # B x 1 x len

        # keys : B x len x emb_dim
        outputs = tf.matmul(outputs, keys)  # B x 1 x dim
        outputs = tf.squeeze(outputs, axis=1)

        return outputs

```

**3）DNN层**

​		到了这层，是对拼接好的整体向量进行一个深度神经网络的操作，最后得到目标输出值。		



## 3、思考

​		在本代码中只实现了历史id的embedding和query item id 的embedding的交互，如果想要引入更多的side- information进行交互的话，代码应该如何修改？