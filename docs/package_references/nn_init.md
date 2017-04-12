# torch.nn.init

```python
torch.nn.init.uniform(tensor, a=0, b=1)
```

从均匀分布U(a, b)中生成值，填充输入的张量或变量

**参数：**

- **tensor** - n维的torch.Tensor
- **a** -均匀分布的下界
- **b** -均匀分布的上界

**例子**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.uniform(w)
```

```python
torch.nn.init.normal(tensor, mean=0, std=1)
```

从给定均值和标准差的正态分布中生成值，填充输入的张量或变量

**参数：**

- **tensor** – n维的torch.Tensor
- **mean** – 正态分布的均值
- **std** – 正态分布的标准差

**例子**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.normal(w)
```

```python
torch.nn.init.constant(tensor, val)
```

用*val*的值填充输入的张量或向量

**参数：**

- **tensor** – n维的torch.Tensor
- **val** – 用来填充张量的值

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.constant(w)
```

```python
torch.nn.init.xavier_uniform(tensor, gain=1)
```

根据Glorot, X.和Bengio, Y.在“Understanding the difficulty of training deep feedforward neural networks”中描述的方法，用一个均匀分布生成值，填充输入的张量或变量。结果张量中的值采样自U(-a, a)，其中a = gain * sqrt(2/(fan_in + fan_out)) * sqrt(3)。

**参数：**

- **tensor** – n维的torch.Tensor
- **gain** - 可选的缩放因子

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.xavier_uniform(w, gain=math.sqrt(2.0))
```

```python
torch.nn.init.xavier_normal(tensor, gain=1)
```

根据Glorot, X.和Bengio, Y.在“Understanding the difficulty of training deep feedforward neural networks”中描述的方法，用一个正态分布生成值，填充输入的张量或变量。结果张量中的值采样自均值为0，标准差为gain * sqrt(2/(fan_in + fan_out))的正态分布。

**参数：**

- **tensor** – n维的torch.Tensor
- **gain** - 可选的缩放因子

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.xavier_normal(w)
```

```python
torch.nn.init.kaiming_uniform(tensor, a=0, mode='fan_in')
```

根据He, K等人在“Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification”中描述的方法，用一个均匀分布生成值，填充输入的张量或变量。结果张量中的值采样自U(-bound, bound)，其中bound = sqrt(2/((1 + a^2) * fan_in)) * sqrt(3)。

**参数：**

- **tensor** – n维的torch.Tensor
- **a** -这层之后使用的rectifier的斜率系数（ReLU的默认值为0）
- **mode** -可以为“fan_in”（默认）或“fan_out”。“fan_in”保留前向传播时权值方差的量级，“fan_out”保留反向传播时的量级。

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.kaiming_uniform(w, mode='fan_in')
```

```python
torch.nn.init.kaiming_normal(tensor, a=0, mode='fan_in')
```

根据He, K等人在“Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification”中描述的方法，用一个正态分布生成值，填充输入的张量或变量。结果张量中的值采样自均值为0，标准差为sqrt( 2/((1 + a^2) * fan_in))的正态分布。

**参数：**

- **tensor** – n维的torch.Tensor
- **a** -这层之后使用的rectifier的斜率系数（ReLU的默认值为0）
- **mode** -可以为“fan_in”（默认）或“fan_out”。“fan_in”保留前向传播时权值方差的量级，“fan_out”保留反向传播时的量级。

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.kaiming_normal(w, mode='fan_out')
```

```python
torch.nn.init.orthogonal(tensor, gain=1)
```

用（半）正交矩阵填充输入的张量或变量。输入张量必须至少是2维的，对于更高维度的张量，超出的维度会被展平，视作行等于第一个维度，列等于稀疏矩阵乘积的2维表示。其中非零元素生成自均值为0，标准差为std的正态分布。

参考：Saxe, A等人的“Exact solutions to the nonlinear dynamics of learning in deep linear neural networks”

**参数：**

- **tensor** – n维的torch.Tensor，其中n>=2
- **gain** -可选

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.orthogonal(w)
```

```python
torch.nn.init.sparse(tensor, sparsity, std=0.01)
```

将2维的输入张量或变量当做稀疏矩阵填充，其中非零元素根据一个均值为0，标准差为std的正态分布生成。

**参数：**

- **tensor** – n维的torch.Tensor
- **sparsity** -每列中需要被设置成零的元素比例
- **std** -用于生成非零值的正态分布的标准差

**例子：**
```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.sparse(w, sparsity=0.1)
```