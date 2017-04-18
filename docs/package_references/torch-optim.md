# torch.optim

`torch.optim`是一个实现了各种优化算法的库。大部分常用的方法得到支持，并且接口具备足够的通用性，使得未来能够集成更加复杂的方法。

## 如何使用optimizer
为了使用`torch.optim`，你需要构建一个optimizer对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。

### 构建
为了构建一个`Optimizer`，你需要给它一个包含了需要优化的参数（必须都是`Variable`对象）的iterable。然后，你可以设置optimizer的参
数选项，比如学习率，权重衰减，等等。

例子：
```python
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)
```

### 为每个参数单独设置选项
`Optimizer`也支持为每个参数单独设置选项。若想这么做，不要直接传入`Variable`的iterable，而是传入`dict`的iterable。每一个dict都分别定
义了一组参数，并且包含一个`param`键，这个键对应参数的列表。其他的键应该optimizer所接受的其他参数的关键字相匹配，并且会被用于对这组参数的
优化。

**`注意：`**

你仍然能够传递选项作为关键字参数。在未重写这些选项的组中，它们会被用作默认值。当你只想改动一个参数组的选项，但其他参数组的选项不变时，这是
非常有用的。

例如，当我们想指定每一层的学习率时，这是非常有用的：

```python
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```

这意味着`model.base`的参数将会使用`1e-2`的学习率，`model.classifier`的参数将会使用`1e-3`的学习率，并且`0.9`的momentum将会被用于所
有的参数。

### 进行一次优化
所有的optimizer都实现了`step()`方法，这个方法会更新所有的参数。它能按两种方式来使用：

**`optimizer.step()`**

这是大多数optimizer所支持的简化版本。一旦梯度被如`backward()`之类的函数计算好后，我们就可以调用这个函数。

例子

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

**`optimizer.step(closure)`**

一些优化算法例如Conjugate Gradient和LBFGS需要重复多次计算函数，因此你需要传入一个闭包去允许它们重新计算你的模型。这个闭包应当清空梯度，
计算损失，然后返回。

例子：

```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```

## 算法

### class torch.optim.Optimizer(params, defaults) [source]
Base class for all optimizers.

**参数：**

* params (iterable) —— `Variable` 或者 `dict`的iterable。指定了什么参数应当被优化。
* defaults —— (dict)：包含了优化选项默认值的字典（一个参数组没有指定的参数选项将会使用默认值）。

#### load_state_dict(state_dict) [source]
加载optimizer状态

**参数：**

state_dict (dict) —— optimizer的状态。应当是一个调用`state_dict()`所返回的对象。

#### state_dict() [source]
以`dict`返回optimizer的状态。

它包含两项。


