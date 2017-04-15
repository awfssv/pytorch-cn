# torch.nn

## 线性层

```python
class torch.nn.Linear(in_features, out_features, bias=True)
```

对输入数据做线性变换：\\(y = Ax + b\\)

**参数：**

- **in_features** - 每个输入样本的大小
- **out_features** - 每个输出样本的大小
- **bias** - 若设置为False，这层不会学习偏置。默认值：True

**形状：**

- **输入:** \\((N, in\\_features)\\)
- **输出：** \\((N, out\\_features)\\)

**变量：**

- **weight** -形状为(out_features x in_features)的模块中可学习的权值
- **bias** -形状为(out_features)的模块中可学习的偏置

**例子：**

```python
>>> m = nn.Linear(20, 30)
>>> input = autograd.Variable(torch.randn(128, 20))
>>> output = m(input)
>>> print(output.size())
```

## Dropout层

``` python
class torch.nn.Dropout(p=0.5, inplace=False)
```

随机将输入张量中部分元素设置为0。对于每次前向调用，被置0的元素都是随机的。

**参数：**

- **p** - 将元素置0的概率。默认值：0.5
- **in-place** - 若设置为True，会在原地执行操作。默认值：False

**形状：**

- **输入：** 任意。输入可以为任意形状。
- **输出：** 相同。输出和输入形状相同。

**例子：**

```python
>>> m = nn.Dropout(p=0.2)
>>> input = autograd.Variable(torch.randn(20, 16))
>>> output = m(input)
```

``` python
class torch.nn.Dropout2d(p=0.5, inplace=False)
```
随机将输入张量中整个通道设置为0。对于每次前向调用，被置0的通道都是随机的。

*通常输入来自Conv2d模块。*

像在论文[Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)，如果特征图中相邻像素是强相关的（在前几层卷积层很常见），那么iid dropout不会归一化激活，而只会降低学习率。

在这种情形，`nn.Dropout2d()`可以提高特征图之间的独立程度，所以应该使用它。

**参数：**

- **p**(*[float](), optional*) - 将元素置0的概率。默认值：0.5
- **in-place**(*[bool,]() optional*) - 若设置为True，会在原地执行操作。默认值：False

**形状：**

- **输入：**  \\((N, C, H, W)\\)
- **输出：**  \\((N, C, H, W)\\)（与输入形状相同）

**例子：**

``` python
>>> m = nn.Dropout2d(p=0.2)
>>> input = autograd.Variable(torch.randn(20, 16, 32, 32))
>>> output = m(input)
```
``` python
class torch.nn.Dropout3d(p=0.5, inplace=False)
```

随机将输入张量中整个通道设置为0。对于每次前向调用，被置0的通道都是随机的。

*通常输入来自Conv3d模块。*

像在论文[Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)，如果特征图中相邻像素是强相关的（在前几层卷积层很常见），那么iid dropout不会归一化激活，而只会降低学习率。

在这种情形，`nn.Dropout3d()`可以提高特征图之间的独立程度，所以应该使用它。

**参数：**

- **p**(*[float](), optional*) - 将元素置0的概率。默认值：0.5
- **in-place**(*[bool,]() optional*) - 若设置为True，会在原地执行操作。默认值：False

**形状：**

- **输入：**  \\(N, C, D, H, W)\\)
- **输出：**  \\((N, C, D, H, W)\\)（与输入形状相同）

**例子：**

``` python
>>> m = nn.Dropout3d(p=0.2)
>>> input = autograd.Variable(torch.randn(20, 16, 4, 32, 32))
>>> output = m(input)
```

##稀疏层

```python
class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
```

一个保存了固定字典和大小的简单查找表。

这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。

**参数：**

- **num_embeddings** (*[int]()*) - 嵌入字典的大小
- **embedding_dim** (*[int]()*) - 每个嵌入向量的大小
- **padding_idx ** (*[int](), optional*) - 如果提供的话，输出遇到此下标时用零填充
- **max_norm** (*[float](), optional*) - 如果提供的话，会重新归一化词嵌入，使它们的范数小于提供的值
-  **norm_type** (*[float](), optional*) - 对于max_norm选项计算p范数时的p 
-  **scale_grad_by_freq** (*boolean, optional*) - 如果提供的话，会根据字典中单词频率缩放梯度

**变量：**

- **weight (*[Tensor](http://pytorch.org/docs/tensors.html#torch.Tensor)*) ** -形状为(num_embeddings, embedding_dim)的模块中可学习的权值

**形状：**

- **输入：** - LongTensor *(N, W)*, N = mini-batch, W = 每个mini-batch中提取的下标数
- **输出：** - *(N, W, embedding_dim)*

**例子：**

```python
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
>>> embedding(input)

Variable containing:
(0 ,.,.) =
 -1.0822  1.2522  0.2434
  0.8393 -0.6062 -0.3348
  0.6597  0.0350  0.0837
  0.5521  0.9447  0.0498

(1 ,.,.) =
  0.6597  0.0350  0.0837
 -0.1527  0.0877  0.4260
  0.8393 -0.6062 -0.3348
 -0.8738 -0.9054  0.4281
[torch.FloatTensor of size 2x4x3]

>>> # example with padding_idx
>>> embedding = nn.Embedding(10, 3, padding_idx=0)
>>> input = Variable(torch.LongTensor([[0,2,0,5]]))
>>> embedding(input)

Variable containing:
(0 ,.,.) =
  0.0000  0.0000  0.0000
  0.3452  0.4937 -0.9361
  0.0000  0.0000  0.0000
  0.0706 -2.1962 -0.6276
[torch.FloatTensor of size 1x4x3]
```

## 距离函数

``` python
class torch.nn.PairwiseDistance(p=2, eps=1e-06)
```
按批计算向量v1, v2之间的距离：


$$\Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}$$


**参数：**

- **x** (*Tensor*):  包含两个输入batch的张量
- **p** (real): 范数次数，默认值：2

**形状：**

- **输入：** - \\((N, D)\\)，其中D=向量维数
- **输出：** - \\((N, 1)\\)

```python
>>> pdist = nn.PairwiseDistance(2)
>>> input1 = autograd.Variable(torch.randn(100, 128))
>>> input2 = autograd.Variable(torch.randn(100, 128))
>>> output = pdist(input1, input2)
```