# torch.nn.functional
## Convolution 函数
```python
torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```
对几个输入平面组成的输入信号应用1D卷积。

有关详细信息和输出形状，请参见`Conv1d`。

**参数：**
- **input** – 输入张量的形状 (minibatch x in_channels x iW)
- **weight** – 过滤器的形状 (out_channels, in_channels, kW)
- **bias** – 可选偏置的形状 (out_channels)
- **stride** – 卷积核的步长，默认为1

**例子：**
```python
>>> filters = autograd.Variable(torch.randn(33, 16, 3))
>>> inputs = autograd.Variable(torch.randn(20, 16, 50))
>>> F.conv1d(inputs, filters)
```

```python
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```
对几个输入平面组成的输入信号应用2D卷积。

有关详细信息和输出形状，请参见`Conv2d`。

**参数：**
- **input** – 输入张量 (minibatch x in_channels x iH x iW)
- **weight** – 过滤器张量 (out_channels, in_channels/groups, kH, kW)
- **bias** – 可选偏置张量 (out_channels)
- **stride** – 卷积核的步长，可以是单个数字或一个元组 (sh x sw)。默认为1
- **padding** – 输入上隐含零填充。可以是单个数字或元组。 默认值：0
- **groups** – 将输入分成组，in_channels应该被组数除尽

**例子：**
```python
>>> # With square kernels and equal stride
>>> filters = autograd.Variable(torch.randn(8,4,3,3))
>>> inputs = autograd.Variable(torch.randn(1,4,5,5))
>>> F.conv2d(inputs, filters, padding=1)
```

```python
torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```
对几个输入平面组成的输入信号应用3D卷积。

有关详细信息和输出形状，请参见`Conv3d`。

**参数：**
- **input** – 输入张量的形状 (minibatch x in_channels x iT x iH x iW)
- **weight** – 过滤器张量的形状 (out_channels, in_channels, kT, kH, kW)
- **bias** – 可选偏置张量的形状 (out_channels)
- **stride** – 卷积核的步长，可以是单个数字或一个元组 (sh x sw)。默认为1
- **padding** – 输入上隐含零填充。可以是单个数字或元组。 默认值：0

**例子：**
```python
>>> filters = autograd.Variable(torch.randn(33, 16, 3, 3, 3))
>>> inputs = autograd.Variable(torch.randn(20, 16, 50, 10, 20))
>>> F.conv3d(inputs, filters)
```

```python
torch.nn.functional.conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1)
```





## 线性函数

```python
torch.nn.functional.linear(input, weight, bias=None)
```

## Dropout 函数

```python
torch.nn.functional.dropout(input, p=0.5, training=False, inplace=False)
```

## 距离函数（Distance functions）

```python
torch.nn.functional.pairwise_distance(x1, x2, p=2, eps=1e-06)
```

计算向量v1、v2之间的距离（成次或者成对，意思是可以计算多个，可以参看后面的参数）
$$
\left \| x \right \|_{p}:=\left ( \sum_{i=1}^{N}\left | x_{i}^{p} \right | \right )^{1/p}
$$
**参数：**

* x1:第一个输入的张量  
* x2:第二个输入的张量  
* p:矩阵范数的维度。默认值是2，即二范数。

**规格：**

- 输入:(N,D)其中D等于向量的维度
- 输出:(N,1)

**例子：**

```python
>>> input1 = autograd.Variable(torch.randn(100, 128))
>>> input2 = autograd.Variable(torch.randn(100, 128))
>>> output = F.pairwise_distance(input1, input2, p=2)
>>> output.backward()
```

