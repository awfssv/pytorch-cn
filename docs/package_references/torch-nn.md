# torch.nn




## Recurrent layers
### class torch.nn.RNN(* args, ** kwargs)[source]

将一个多层的 `Elman RNN`，激活函数为`tanh`或者`ReLU`，用于输入序列。

对输入序列中每个元素，`RNN`每层的计算公式为
$$
h_t=tanh(w_{ih}* x_t+b_{ih}+w_{hh}* h_{t-1}+b_{hh})
$$
$h_t$是时刻$t$的隐状态。 $x_t$是上一层时刻$t$的隐状态，或者是第一层在时刻$t$的输入。如果`nonlinearity='relu'`,那么将使用`relu`代替`tanh`作为激活函数。

参数说明:

- input_size – 输入`x`的特征数量。

- hidden_size – 隐层的特征数量。

- num_layers – RNN的层数。

- nonlinearity – 指定非线性函数使用`tanh`还是`relu`。默认是`tanh`。

- bias – 如果是`False`，那么RNN层就不会使用偏置权重 $b_ih$和$b_hh$,默认是`True`

- batch_first – 如果`True`的话，那么输入`Tensor`的shape应该是[batch_size, time_step, feature],输出也是这样。
- dropout – 如果值非零，那么除了最后一层外，其它层的输出都会套上一个`dropout`层。

- bidirectional – 如果`True`，将会变成一个双向`RNN`，默认为`False`。


`RNN`的输入：
**(input, h_0)**
- input (seq_len, batch, input_size): 保存输入序列特征的`tensor`。`input`可以是被填充的变长的序列。细节请看`torch.nn.utils.rnn.pack_padded_sequence()`

- h_0 (num_layers * num_directions, batch, hidden_size): 保存着初始隐状态的`tensor`

`RNN`的输出：
**(output, h_n)**

- output (seq_len, batch, hidden_size * num_directions): 保存着`RNN`最后一层的输出特征。如果输入是被填充过的序列，那么输出也是被填充的序列。
- h_n (num_layers * num_directions, batch, hidden_size): 保存着最后一个时刻隐状态。

`RNN`模型参数:

- weight_ih_l[k] – 第`k`层的 `input-hidden` 权重， 可学习，形状是`(input_size x hidden_size)`。

- weight_hh_l[k] – 第`k`层的 `hidden-hidden` 权重， 可学习，形状是`(hidden_size x hidden_size)`

- bias_ih_l[k] – 第`k`层的 `input-hidden` 偏置， 可学习，形状是`(hidden_size)`

- bias_hh_l[k] – 第`k`层的 `hidden-hidden` 偏置， 可学习，形状是`(hidden_size)`

示例：
```python
rnn = nn.RNN(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)
```
### class torch.nn.LSTM(* args, ** kwargs)[source]

将一个多层的 `(LSTM)` 应用到输入序列。

对输入序列的每个元素，`LSTM`的每层都会执行以下计算：
$$
\begin{aligned}
i_t &= sigmoid(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi}) \\
f_t &= sigmoid(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf}) \\
o_t &= sigmoid(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho})\\
g_t &= tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg})\\
c_t &= f_t*c_{t-1}+i_t*g_t\\
h_t &= o_t*tanh(c_t)
\end{aligned}
$$
$h_t$是时刻$t$的隐状态,$c_t$是时刻$t$的细胞状态，$x_t$是上一层的在时刻$t$的隐状态或者是第一层在时刻$t$的输入。$i_t, f_t, g_t, o_t$ 分别代表 输入门，遗忘门，细胞和输出门。

参数说明:

- input_size – 输入的特征维度

- hidden_size – 隐状态的特征维度

- num_layers – 层数（和时序展开要区分开）

- bias – 如果为`False`，那么`LSTM`将不会使用$b_{ih},b_{hh}$，默认为`True`。

- batch_first – 如果为`True`，那么输入和输出`Tensor`的形状为`(batch, seq, feature)`

- dropout – 如果非零的话，将会在`RNN`的输出上加个`dropout`，最后一层除外。

- bidirectional – 如果为`True`，将会变成一个双向`RNN`，默认为`False`。

`LSTM`输入:
input, (h_0, c_0)

- input (seq_len, batch, input_size): 包含输入序列特征的`Tensor`。也可以是`packed variable` ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])

- h_0 (num_layers * num_directions, batch, hidden_size):保存着`batch`中每个元素的初始化隐状态的`Tensor`

- c_0 (num_layers * num_directions, batch, hidden_size): 保存着`batch`中每个元素的初始化细胞状态的`Tensor`

`LSTM`输出
output, (h_n, c_n)

- output (seq_len, batch, hidden_size * num_directions): 保存`RNN`最后一层的输出的`Tensor`。 如果输入是`torch.nn.utils.rnn.PackedSequence`，那么输出也是`torch.nn.utils.rnn.PackedSequence`。

- h_n (num_layers * num_directions, batch, hidden_size): `Tensor`，保存着`RNN`最后一个时间步的隐状态。

- c_n (num_layers * num_directions, batch, hidden_size): `Tensor`，保存着`RNN`最后一个时间步的细胞状态。

`LSTM`模型参数:

- weight_ih_l[k] – 第`k`层可学习的`input-hidden`权重($W_{ii}|W_{if}|W_{ig}|W_{io}$)，形状为`(input_size x 4*hidden_size)`

- weight_hh_l[k] – 第`k`层可学习的`hidden-hidden`权重($W_{hi}|W_{hf}|W_{hg}|W_{ho}$)，形状为`(hidden_size x 4*hidden_size)`。

- bias_ih_l[k] – 第`k`层可学习的`input-hidden`偏置($b_{ii}|b_{if}|b_{ig}|b_{io}$)，形状为`( 4*hidden_size)`

- bias_hh_l[k] – 第`k`层可学习的`hidden-hidden`偏置($b_{hi}|b_{hf}|b_{hg}|b_{ho}$)，形状为`( 4*hidden_size)`。
示例:
```python
lstm = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = lstm(input, (h0, c0))
```

### class torch.nn.GRU(* args, ** kwargs)[source]

将一个多层的`GRU`用于输入序列。

对输入序列中的每个元素，每层进行了一下计算：

$$
\begin{aligned}
r_t&=sigmoid(W_{ir}x_t+b_{ir}+W_{hr}h_{(t-1)}+b_{hr})\\
i_t&=sigmoid(W_{ii}x_t+b_{ii}+W_{hi}h_{(t-1)}+b_{hi})\\
n_t&=tanh(W_{in}x_t+b_{in}+rt*(W_{hn}h_{(t-1)}+b_{hn}))\\
h_t&=(1-i_t)* nt+i_t*h(t-1)
\end{aligned}
$$
$h_t$是是时间$t$的上的隐状态，$x_t$是前一层$t$时刻的隐状态或者是第一层的$t$时刻的输入，$r_t, i_t, n_t$分别是重置门，输入门和新门。

参数说明：
- input_size – 期望的输入$x$的特征值的维度
- hidden_size – 隐状态的维度
- num_layers – `RNN`的层数。
- bias – 如果为`False`，那么`RNN`层将不会使用`bias`，默认为`True`。
- batch_first – 如果为`True`的话，那么输入和输出的`tensor`的形状是`(batch, seq, feature)`。
- dropout –  如果非零的话，将会在`RNN`的输出上加个`dropout`，最后一层除外。
- bidirectional – 如果为`True`，将会变成一个双向`RNN`，默认为`False`。

输入：
input, h_0

- input (seq_len, batch, input_size):  包含输入序列特征的`Tensor`。也可以是`packed variable` ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])。

- h_0 (num_layers * num_directions, batch, hidden_size):保存着`batch`中每个元素的初始化隐状态的`Tensor`

输出：
output, h_n

- output (seq_len, batch, hidden_size * num_directions): ten保存`RNN`最后一层的输出的`Tensor`。 如果输入是`torch.nn.utils.rnn.PackedSequence`，那么输出也是`torch.nn.utils.rnn.PackedSequence`。

- h_n (num_layers * num_directions, batch, hidden_size): `Tensor`，保存着`RNN`最后一个时间步的隐状态。

变量：

- weight_ih_l[k] – 第`k`层可学习的`input-hidden`权重($W_{ir}|W_{ii}|W_{in}$)，形状为`(input_size x 3*hidden_size)`

- weight_hh_l[k] – 第`k`层可学习的`hidden-hidden`权重($W_{hr}|W_{hi}|W_{hn}$)，形状为`(hidden_size x 3*hidden_size)`。

- bias_ih_l[k] – 第`k`层可学习的`input-hidden`偏置($b_{ir}|b_{ii}|b_{in}$)，形状为`( 3*hidden_size)`

- bias_hh_l[k] – 第`k`层可学习的`hidden-hidden`偏置($b_{hr}|b_{hi}|b_{hn}$)，形状为`( 3*hidden_size)`。


例子：
```python
 rnn = nn.GRU(10, 20, 2)
 input = Variable(torch.randn(5, 3, 10))
 h0 = Variable(torch.randn(2, 3, 20))
 output, hn = rnn(input, h0)
```

### class torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')[source]

一个 `Elan RNN cell`，激活函数是`tanh`或`ReLU`，用于输入序列。
将一个多层的 `Elman RNNCell`，激活函数为`tanh`或者`ReLU`，用于输入序列。
$$
h'=tanh(w_{ih}* x+b_{ih}+w_{hh}* h+b_{hh})
$$
如果`nonlinearity=relu`，那么将会使用`ReLU`来代替`tanh`。

参数：

- input_size – 输入$x$，特征的维度。

- hidden_size – 隐状态特征的维度。

- bias – 如果为`False`，`RNN cell`中将不会加入`bias`，默认为`True`。

- nonlinearity – 用于选择非线性激活函数 [`tanh`|`relu`]. 默认值为： `tanh`

输入：
input, hidden

- input (batch, input_size): 包含输入特征的`tensor`。

- hidden (batch, hidden_size): 保存着初始隐状态值的`tensor`。

输出： h’

- h’ (batch, hidden_size):下一个时刻的隐状态。

变量：

- weight_ih –  `input-hidden` 权重， 可学习，形状是`(input_size x hidden_size)`。

- weight_hh –  `hidden-hidden` 权重， 可学习，形状是`(hidden_size x hidden_size)`

- bias_ih –  `input-hidden` 偏置， 可学习，形状是`(hidden_size)`

- bias_hh –  `hidden-hidden` 偏置， 可学习，形状是`(hidden_size)`

例子：

```python
rnn = nn.RNNCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
   hx = rnn(input[i], hx)
   output.append(hx)
```

### class torch.nn.LSTMCell(input_size, hidden_size, bias=True)[source]

`LSTM cell`。
$$
\begin{aligned}
i &= sigmoid(W_{ii}x+b_{ii}+W_{hi}h+b_{hi}) \\
f &= sigmoid(W_{if}x+b_{if}+W_{hf}h+b_{hf}) \\
o &= sigmoid(W_{io}x+b_{io}+W_{ho}h+b_{ho})\\
g &= tanh(W_{ig}x+b_{ig}+W_{hg}h+b_{hg})\\
c' &= f_t*c_{t-1}+i_t*g_t\\
h' &= o_t*tanh(c')
\end{aligned}
$$

参数：

- input_size – 输入的特征维度。
- hidden_size – 隐状态的维度。
- bias – 如果为`False`，那么将不会使用`bias`。默认为`True`。

`LSTM`输入:
input, (h_0, c_0)

- input (seq_len, batch, input_size): 包含输入序列特征的`Tensor`。也可以是`packed variable` ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])

- h_0 ( batch, hidden_size):保存着`batch`中每个元素的初始化隐状态的`Tensor`

- c_0 (batch, hidden_size): 保存着`batch`中每个元素的初始化细胞状态的`Tensor`

输出：
h_1, c_1

- h_1 (batch, hidden_size): 下一个时刻的隐状态。
- c_1 (batch, hidden_size): 下一个时刻的细胞状态。

`LSTM`模型参数:

- weight_ih – `input-hidden`权重($W_{ii}|W_{if}|W_{ig}|W_{io}$)，形状为`(input_size x 4*hidden_size)`

- weight_hh – `hidden-hidden`权重($W_{hi}|W_{hf}|W_{hg}|W_{ho}$)，形状为`(hidden_size x 4*hidden_size)`。

- bias_ih – `input-hidden`偏置($b_{ii}|b_{if}|b_{ig}|b_{io}$)，形状为`( 4*hidden_size)`

- bias_hh – `hidden-hidden`偏置($b_{hi}|b_{hf}|b_{hg}|b_{ho}$)，形状为`( 4*hidden_size)`。

Examples:
```python
rnn = nn.LSTMCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
cx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
   hx, cx = rnn(input[i], (hx, cx))
   output.append(hx)
```

### class torch.nn.GRUCell(input_size, hidden_size, bias=True)[source]

一个`GRU cell`。
$$
\begin{aligned}
r&=sigmoid(W_{ir}x+b_{ir}+W_{hr}h+b_{hr})\\
i&=sigmoid(W_{ii}x+b_{ii}+W_{hi}h+b_{hi})\\
n&=tanh(W_{in}x+b_{in}+r*(W_{hn}h+b_{hn}))\\
h'&=(1-i)* n+i*h
\end{aligned}
$$

参数说明：
- input_size – 期望的输入$x$的特征值的维度
- hidden_size – 隐状态的维度
- bias – 如果为`False`，那么`RNN`层将不会使用`bias`，默认为`True`。

输入：
input, h_0

- input (batch, input_size):  包含输入特征的`Tensor`

- h_0 (batch, hidden_size):保存着`batch`中每个元素的初始化隐状态的`Tensor`

输出：
h_1

- h_1 (batch, hidden_size): `Tensor`，保存着`RNN`下一个时刻的隐状态。

变量：

- weight_ih – `input-hidden`权重($W_{ir}|W_{ii}|W_{in}$)，形状为`(input_size x 3*hidden_size)`

- weight_hh – `hidden-hidden`权重($W_{hr}|W_{hi}|W_{hn}$)，形状为`(hidden_size x 3*hidden_size)`。

- bias_ih – `input-hidden`偏置($b_{ir}|b_{ii}|b_{in}$)，形状为`( 3*hidden_size)`

- bias_hh – `hidden-hidden`偏置($b_{hr}|b_{hi}|b_{hn}$)，形状为`( 3*hidden_size)`。

例子：
```python
rnn = nn.GRUCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
   hx = rnn(input[i], hx)
   output.append(hx)
```
## Linear layers

## Dropout layers

## Sparse layers

## Distance functions

## Loss functions
基本用法：
```python
criterion = LossCriterion() #构造函数有自己的参数
loss = criterion(x, y) #调用标准时也有参数
```
计算出来的结果已经对`mini-batch`取了平均。
### class torch.nn.L1Loss(size_average=True)[source]
创建一个衡量输入`x`(`模型预测输出`)和目标`y`之间差的绝对值的平均值的标准。
$$
loss(x,y)=1/n\sum|x_i-y_i|
$$

- `x` 和 `y` 可以是任意形状，每个包含`n`个元素。

- 对`n`个元素对应的差值的绝对值求和，得出来的结果除以`n`。

- 如果在创建`L1Loss`实例的时候在构造函数中传入`size_average=False`，那么求出来的绝对值的和将不会除以`n`


### class torch.nn.MSELoss(size_average=True)[source]
创建一个衡量输入`x`(`模型预测输出`)和目标`y`之间均方误差标准。
$$
loss(x,y)=1/n\sum(x_i-y_i)^2
$$

- `x` 和 `y` 可以是任意形状，每个包含`n`个元素。

- 对`n`个元素对应的差值的绝对值求和，得出来的结果除以`n`。

- 如果在创建`MSELoss`实例的时候在构造函数中传入`size_average=False`，那么求出来的平方和将不会除以`n`

### class torch.nn.CrossEntropyLoss(weight=None, size_average=True)[source]
此标准将`LogSoftMax`和`NLLLoss`集成到一个类中。

当训练一个多类分类器的时候，这个方法是十分有用的。

- weight(tensor): `1-D` tensor，`n`个元素，分别代表`n`类的权重，如果你的训练样本很不均衡的话，是非常有用的。默认值为None。

调用时参数：

- input : 包含每个类的得分，`2-D` tensor,`shape`为 `batch*n`

- target: 大小为 `n` 的 `1—D` `tensor`，包含类别的索引(`0到 n-1`)。


Loss可以表述为以下形式：
$$
\begin{aligned}
loss(x, class) &= -\text{log}\frac{exp(x[class])}{\sum_j exp(x[j]))}\\
               &= -x[class] + log(\sum_j exp(x[j]))
\end{aligned}
$$
当`weight`参数被指定的时候，`loss`的计算公式变为：
$$
loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))
$$
计算出的`loss`对`mini-batch`的大小取了平均。

形状(`shape`)：

- Input: (N,C) `C` 是类别的数量

- Target: (N) `N`是`mini-batch`的大小，0 <= targets[i] <= C-1

### class torch.nn.NLLLoss(weight=None, size_average=True)[source]
负的`log likelihood loss`损失。用于训练一个`n`类分类器。

如果提供的话，`weight`参数应该是一个`1-D`tensor，里面的值对应类别的权重。当你的训练集样本不均衡的话，使用这个参数是非常有用的。

输入是一个包含类别`log-probabilities`的`2-D` tensor，形状是`（mini-batch， n）`

可以通过在最后一层加`LogSoftmax`来获得类别的`log-probabilities`。

如果您不想增加一个额外层的话，您可以使用`CrossEntropyLoss`。

此`loss`期望的`target`是类别的索引 (0 to N-1, where N = number of classes)

此`loss`可以被表示如下：
$$
loss(x, class) = -x[class]
$$
如果`weights`参数被指定的话，`loss`可以表示如下：
$$
loss(x, class) = -weights[class] * x[class]
$$
参数说明：

- weight (Tensor, optional) – 手动指定每个类别的权重。如果给定的话，必须是长度为`nclasses`

- size_average (bool, optional) – 默认情况下，会计算`mini-batch``loss`的平均值。然而，如果`size_average=False`那么将会把`mini-batch`中所有样本的`loss`累加起来。

形状:

- Input: (N,C) , `C`是类别的个数

- Target: (N) ， `target`中每个值的大小满足 `0 <= targets[i] <= C-1`

例子:
```python
 m = nn.LogSoftmax()
 loss = nn.NLLLoss()
 # input is of size nBatch x nClasses = 3 x 5
 input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
 # each element in target has to have 0 <= value < nclasses
 target = autograd.Variable(torch.LongTensor([1, 0, 4]))
 output = loss(m(input), target)
 output.backward()
```

### class torch.nn.NLLLoss2d(weight=None, size_average=True)[source]

对于图片的 `negative log likehood loss`。计算每个像素的 `NLL loss`。

参数说明：

- weight (Tensor, optional) – 用来作为每类的权重，如果提供的话，必须为`1-D`tensor，大小为`C`：类别的个数。

- size_average – 默认情况下，会计算 `mini-batch` loss均值。如果设置为 `False` 的话，将会累加`mini-batch`中所有样本的`loss`值。默认值：`True`。

形状：

- Input: (N,C,H,W)  `C` 类的数量

- Target: (N,H,W) where each value is 0 <= targets[i] <= C-1

例子：
```python
 m = nn.Conv2d(16, 32, (3, 3)).float()
 loss = nn.NLLLoss2d()
 # input is of size nBatch x nClasses x height x width
 input = autograd.Variable(torch.randn(3, 16, 10, 10))
 # each element in target has to have 0 <= value < nclasses
 target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
 output = loss(m(input), target)
 output.backward()
```
### class torch.nn.KLDivLoss(weight=None, size_average=True)[source]

计算 KL 散度损失。

KL散度常用来描述两个分布的距离，并在输出分布的空间上执行直接回归是有用的。

与`NLLLoss`一样，给定的输入应该是`log-probabilities`。然而。和`NLLLoss`不同的是，`input`不限于`2-D` tensor，因为此标准是基于`element`的。

`target` 应该和 `input`的形状相同。

此loss可以表示为：
$$
loss(x,target)=\frac{1}{n}\sum_i(target_i*(log(target_i)-x_i))
$$
默认情况下，loss会基于`element`求平均。如果 `size_average=False` `loss` 会被累加起来。

### class torch.nn.BCELoss(weight=None, size_average=True)[source]

计算 `target` 与 `output` 之间的二进制交叉熵。
$$
loss(o,t)=-\frac{1}{n}\sum_i(t[i]* log(o[i])+(1-t[i])* log(1-o[i]))
$$
如果`weight`被指定 ：
$$
loss(o,t)=-\frac{1}{n}\sum_iweights[i]* (t[i]* log(o[i])+(1-t[i])* log(1-o[i]))
$$

这个用于计算 `auto-encoder` 的 `reconstruction error`。注意 0<=target[i]<=1。

默认情况下，loss会基于`element`平均，如果`size_average=False`的话，`loss`会被累加。

### class torch.nn.MarginRankingLoss(margin=0, size_average=True)[source]

创建一个标准，给定输入 $x1$,$x2$两个1-D mini-batch Tensor's，和一个$y$(1-D mini-batch tensor) ,$y$里面的值只能是-1或1。

如果 `y=1`，代表第一个输入的值应该大于第二个输入的值，如果`y=-1`的话，则相反。

`mini-batch`中每个样本的loss的计算公式如下：

$$loss(x, y) = max(0, -y * (x1 - x2) + margin)$$

如果`size_average=True`,那么求出的`loss`将会对`mini-batch`求平均，反之，求出的`loss`会累加。默认情况下，`size_average=True`。

### class torch.nn.HingeEmbeddingLoss(size_average=True)[source]

给定一个输入 $x$(2-D mini-batch tensor)和对应的 标签 $y$ (1-D tensor,1,-1)，此函数用来计算之间的损失值。这个`loss`通常用来测量两个输入是否相似，即：使用L1 成对距离。典型是用在学习非线性 `embedding`或者半监督学习中：

$$
loss(x,y)=\frac{1}{n}\sum_i
\begin{cases}
x_i, &\text if~y_i==1 \\
max(0, margin-x_i), &if ~y_i==-1
\end{cases}
$$
$x$和$y$可以是任意形状，且都有`n`的元素，`loss`的求和操作作用在所有的元素上，然后除以`n`。如果您不想除以`n`的话，可以通过设置`size_average=False`。

`margin`的默认值为1,可以通过构造函数来设置。

### class torch.nn.MultiLabelMarginLoss(size_average=True)[source]
计算多标签分类的 `hinge loss`(`margin-based loss`) ，计算`loss`时需要两个输入： input x(`2-D mini-batch Tensor`)，和 output y(`2-D tensor`表示mini-batch中样本类别的索引)。

$$
loss(x, y) = \frac{1}{x.size(0)}\sum_{i=0,j=0}^{I,J}(max(0, 1 - (x[y[j]] - x[i])))
$$
其中 `I=x.size(0),J=y.size(0)`。对于所有的 `i`和 `j`，满足 $y[j]\neq0, i \neq y[j]$

`x` 和 `y` 必须具有同样的 `size`。

这个标准仅考虑了第一个非零 `y[j] targets`
此标准允许了，对于每个样本来说，可以有多个类别。

### class torch.nn.SmoothL1Loss(size_average=True)[source]
平滑版`L1 loss`。

loss的公式如下：
$$
loss(x, y) = \frac{1}{n}\sum_i
\begin{cases}
0.5*(x_i-y_i)^2, & if~|x_i - y_i| < 1\\
|x_i - y_i| - 0.5,  & otherwise    
\end{cases}
$$
此loss对于异常点的敏感性不如`MSELoss`，而且，在某些情况下防止了梯度爆炸，(参照 `Fast R-CNN`)。这个`loss`有时也被称为 `Huber loss`。

x 和 y 可以是任何包含`n`个元素的tensor。默认情况下，求出来的`loss`会除以`n`，可以通过设置`size_average=True`使loss累加。

### class torch.nn.SoftMarginLoss(size_average=True)[source]

创建一个标准，用来优化2分类的`logistic loss`。输入为 `x`（一个 2-D mini-batch Tensor）和 目标`y`（一个包含1或-1的Tensor）。
$$
loss(x, y) = \frac{1}{x.nelement()}\sum_i (log(1 + exp(-y[i]* x[i])))
$$
如果求出的`loss`不想被平均可以通过设置`size_average=False`。

### class torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=True)[source]

创建一个标准，基于输入x和目标y的 `max-entropy`，优化多标签 `one-versus-all` 的损失。`x`:2-D mini-batch Tensor;`y`:binary 2D Tensor。对每个mini-batch中的样本，对应的loss为：
$$
loss(x, y) = - \frac{1}{x.nElement()}\sum_{i=0}^I y[i]\text{log}\frac{exp(x[i])}{(1 + exp(x[i])}
                      + (1-y[i])\text{log}\frac{1}{1+exp(x[i])}
$$
其中 `I=x.nElement()-1`, $y[i] \in \{0,1\}$，`y` 和 `x`必须要有同样`size`。

### class torch.nn.CosineEmbeddingLoss(margin=0, size_average=True)[source]

给定 输入 `Tensors`，`x1`, `x2` 和一个标签Tensor `y`(元素的值为1或-1)。此标准使用`cosine`距离测量两个输入是否相似，一般用来用来学习非线性`embedding`或者半监督学习。

`margin`应该是-1到1之间的值，建议使用0到0.5。如果没有传入`margin`实参，默认值为0。

每个样本的loss是：
$$
loss(x, y) =
\begin{cases}
1 - cos(x1, x2),              &if~y ==  1 \\
max(0, cos(x1, x2) - margin), &if~y == -1
\end{cases}
$$
如果`size_average=True` 求出的loss会对batch求均值，如果`size_average=False`的话，则会累加`loss`。默认情况`size_average=True`。

### class torch.nn.MultiMarginLoss(p=1, margin=1, weight=None, size_average=True)[source]
用来计算multi-class classification的hinge loss（magin-based loss）。输入是 `x`(2D mini-batch Tensor), `y`(1D Tensor)包含类别的索引， `0 <= y <= x.size(1))`。

对每个mini-batch样本：
$$
loss(x, y) = \frac{1}{x.size(0)}\sum_{i=0}^I(max(0, margin - x[y] + x[i])^p)
$$
其中 `I=x.size(0)` $i\neq y$。
可选择的，如果您不想所有的类拥有同样的权重的话，您可以通过在构造函数中传入`weights`参数来解决这个问题，`weights`是一个1D权重Tensor。

传入weights后，loss函数变为：
$$
loss(x, y) = \frac{1}{x.size(0)}\sum_imax(0, w[y] * (margin - x[y] - x[i]))^p
$$
默认情况下，求出的loss会对mini-batch取平均，可以通过设置`size_average=False`来取消取平均操作。

## Vision layers

### class torch.nn.PixelShuffle(upscale_factor)[source]

将shape为$[N, C*r^2, H, W]$的`Tensor`重新排列为shape为$[N, C, H*r, W*r]$的Tensor。
当使用`stride=1/r` 的sub-pixel卷积的时候，这个方法是非常有用的。

请看paper[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network by Shi et. al (2016)](https://arxiv.org/abs/1609.05158) 获取详细信息。

参数说明：

- upscale_factor (int) – 增加空间分辨率的因子

Shape:

- Input: $[N,C*upscale\_factor^2,H,W$]

- Output: $[N,C,H*upscale\_factor,W*upscale\_factor]$

例子:
```python
>>> ps = nn.PixelShuffle(3)
>>> input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
>>> output = ps(input)
>>> print(output.size())
torch.Size([1, 1, 12, 12])
```


### class torch.nn.UpsamplingNearest2d(size=None, scale_factor=None)[source]

对于多channel 输入 进行 `2-D` 最近邻上采样。

可以通过`size`或者`scale_factor`来指定上采样后的图片大小。

当给定`size`时，`size`的值将会是输出图片的大小。

参数：

- size (tuple, optional) – 一个包含两个整数的元组 (H_out, W_out)指定了输出的长宽
- scale_factor (int, optional) – 长和宽的一个乘子

形状：

- Input: (N,C,H_in,W_in)
- Output: (N,C,H_out,W_out)  Hout=floor(H_in∗scale_factor) Wout=floor(W_in∗scale_factor)

例子：
```python
>>> inp
Variable containing:
(0 ,0 ,.,.) =
  1  2
  3  4
[torch.FloatTensor of size 1x1x2x2]

>>> m = nn.UpsamplingNearest2d(scale_factor=2)
>>> m(inp)
Variable containing:
(0 ,0 ,.,.) =
  1  1  2  2
  1  1  2  2
  3  3  4  4
  3  3  4  4
[torch.FloatTensor of size 1x1x4x4]
```

### class torch.nn.UpsamplingBilinear2d(size=None, scale_factor=None)[source]

对于多channel 输入 进行 `2-D` `bilinear` 上采样。

可以通过`size`或者`scale_factor`来指定上采样后的图片大小。

当给定`size`时，`size`的值将会是输出图片的大小。

参数：

- size (tuple, optional) – 一个包含两个整数的元组 (H_out, W_out)指定了输出的长宽
- scale_factor (int, optional) – 长和宽的一个乘子

形状：

- Input: (N,C,H_in,W_in)
- Output: (N,C,H_out,W_out)  Hout=floor(H_in∗scale_factor) Wout=floor(W_in∗scale_factor)

例子：
```python
>>> inp
Variable containing:
(0 ,0 ,.,.) =
  1  2
  3  4
[torch.FloatTensor of size 1x1x2x2]

>>> m = nn.UpsamplingBilinear2d(scale_factor=2)
>>> m(inp)
Variable containing:
(0 ,0 ,.,.) =
  1.0000  1.3333  1.6667  2.0000
  1.6667  2.0000  2.3333  2.6667
  2.3333  2.6667  3.0000  3.3333
  3.0000  3.3333  3.6667  4.0000
[torch.FloatTensor of size 1x1x4x4]
```
## Multi-GPU layers
### class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)[source]

在模块级别上实现数据并行。

此容器通过将`mini-batch`划分到不同的设备上来实现给定`module`的并行。在`forward`过程中，`module`会在每个设备上都复制一遍，每个副本都会处理部分输入。在`backward`过程中，副本上的梯度会累加到原始`module`上。

batch的大小应该大于所使用的GPU的数量。还应当是GPU个数的整数倍，这样划分出来的每一块都会有相同的样本数量。

请看: [Use nn.DataParallel instead of multiprocessing]()

除了`Tensor`，任何位置参数和关键字参数都可以传到DataParallel中。所有的变量会通过指定的`dim`来划分（默认值为0）。原始类型将会被广播，但是所有的其它类型都会被浅复制。所以如果在模型的`forward`过程中写入的话，将会被损坏。

参数说明：

- module – 要被并行的module
- device_ids – CUDA设备，默认为所有设备。
- output_device – 输出设备（默认为device_ids[0]）

例子：
```python
 net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
 output = net(input_var)
```

## Utilities
工具函数
### torch.nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2)[source]
Clips gradient norm of an iterable of parameters.

正则項的值由所有的梯度计算出来，就像他们连成一个向量一样。梯度被`in-place operation`修改。

参数说明:
- parameters (Iterable[Variable]) – 可迭代的`Variables`，它们的梯度即将被标准化。
- max_norm (float or int) – `clip`后，`gradients` p-norm 值
- norm_type (float or int) – 标准化的类型，p-norm. 可以是`inf` 代表 infinity norm.

[关于norm](https://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/)

返回值:

所有参数的p-norm值。

### torch.nn.utils.rnn.PackedSequence(\_cls, data, batch_sizes)[source]
Holds the data and list of batch_sizes of a packed sequence.

All RNN modules accept packed sequences as inputs.
所有的`RNN`模块都接收这种被包裹后的序列作为它们的输入。

`NOTE：`
这个类的实例不能手动创建。它们只能被 `pack_padded_sequence()` 实例化。

参数说明:

- data (Variable) – 包含打包后序列的`Variable`。

- batch_sizes (list[int]) – 包含 `mini-batch` 中每个序列长度的列表。

#### torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False)[source]
Packs a Variable containing padded sequences of variable length.
把一个包含 填充过的变长序列 `Variable`打包。

输入的形状可以是(T×B×* )。`T`是最长序列长度，`B`是`batch size`，`*`代表任意维度(可以是0)。如果`batch_first=True`的话，那么相应的 `input size` 就是 `(B×T×*)`。

`Variable`中保存的序列，应该按序列长度的长短排序，长的在前，短的在后。即`input[:,0]`代表的是最长的序列，`input[:, B-1]`保存的是最短的序列。

`NOTE：`
只要是维度大于等于2的`input`都可以作为这个函数的参数。你可以用它来打包`labels`，然后用`RNN`的输出和打包后的`labels`来计算`loss`。通过`PackedSequence`对象的`.data`属性可以获取 `Variable`。

参数说明:

- input (Variable) – 变长序列 被填充后的 batch

- lengths (list[int]) – `Variable` 中 每个序列的长度。

- batch_first (bool, optional) – 如果是`True`，input的形状应该是`B*T*size`。

返回值:

一个`PackedSequence` 对象。

### torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False)[source]

填充变长序列的打包的batch
Pads a packed batch of variable length sequences.
这个操作和pack_padded_sequence()是相反的。

返回的Varaible的值的`size`是 `T×B×*`, `T` 是最长序列的长度，`B` 是 batch_size,如果 `batch_first=True`,那么返回值是`B×T×*`。

Batch中的元素将会以它们长度的逆序排列。


参数说明:

- sequence (PackedSequence) – 将要被填充的 batch

- batch_first (bool, optional) – 如果为True，返回的数据的格式为 `B×T×*`。

返回值:
一个tuple，包含被填充后的序列，和batch中序列的长度列表。
