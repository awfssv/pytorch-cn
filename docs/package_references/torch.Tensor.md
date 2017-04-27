# torch.Tensor
`torch.Tensor`是一种包含单一数据类型元素的多维矩阵。

Torch定义了七种CPU tensor类型和八种GPU tensor类型：

| Data tyoe | CPU tensor | GPU tensor|
| :---: | :---: | :---: |
| 32-bit floating point | `torch.floattenor` | `torch.cuda.floattensor` |
| 64-bit floating point | `torch.DoubleTenor` | `torch.cuda.DoubleTensor` |
| 16-bit floating point | N/A | `torch.cuda.HalfTensor` |
| 8-bit integer (unsigned) | `torch.ByteTenor` | `torch.cuda.ByteTensor` |
| 8-bit integer (signed) | `torch.CharTenor` | `torch.cuda.CharTensor` |
| 16-bit integer (signed) | `torch.ShortTenor` | `torch.cuda.ShortTensor` |
| 32-bit integer (signed) | `torch.IntTenor` | `torch.cuda.IntTensor` |
| 64-bit integer (signed) | `torch.LongTenor` | `torch.cuda.LongTensor` |

`torch.Tensor`是默认的tensor类型（`torch.FlaotTensor`）的简称。

一个张量tensor可以从Python的`list`或序列构建：
```python
>>> torch.FloatTensor([[1,2,3],[4,5,6]])
1 2 3
4 5 6
[torch.FloatTensor of size 2x3]
```
一个空张量tensor可以通过规定其大小来构建：

```python
>>> torch.IntTensor(2, 4).zero_()
0 0 0 0
0 0 0 0
[torch.IntTensor of size 2x4]
```

可以用python的indexing和slicing来获取和修改一个张量tensor中的内容：

```python
>>> x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
>>> print(x[1][2])
6.0
>>> x[0][1] = 8
>>> print(x)
 1 8 3
 4 5 6
[torch.FloatTensor of size 2x3]
```

每一个张量tensor都有一个相应的`torch.Storage`用来保存其数据。类tensor提供了多维的！！！！！！！！！！存储视图，并且定义了在数值运算。


>！注意：
会改变tensor的函数操作会用一个下划线后缀来标示。比如，`torch.FloatTensor.abs_()`会在原地计算绝对值，并返回改变后的tensor，而`tensor.FloatTensor.abs()`将会在一个新的tensor中计算结果。


```python
class torch.Tensor
class torch.Tensor(*sizes)
class torch.Tensor(size)
class torch.Tensor(sequence)
class torch.Tensor(ndarray)
class torch.Tensor(tensor)
class torch.Tensor(storage)
```
根据可选择的大小和数据新建一个tensor。
如果没有提供参数，将会返回一个空的零维张量。如果提供了`numpy.ndarray`,`torch.Tensor`或`torch.Storage`，将会返回一个有同样参数的tensor.如果提供了python序列，将会新建一个y与序列相同的tensor。

#### abs() $\rightarrow$ Tensor
请查看`torch.abs()`
#### abs_() $\rightarrow$ Tensor
`abs()`的原地运算形式
#### acos() $\rightarrow$ Tensor
请查看`torch.acos()`
#### acos_() $\rightarrow$ Tensor
`acos()`的原地运算形式
#### add(value)
请查看`torch.add()`
#### add_(value)
`add()`的原地运算形式
#### addbmm(beta=1, mat, alpha=1, batch1, batch2) $\to$ Tensor
请查看`torch.addbmm()`
#### addbmm_(beta=1, mat, alpha=1, batch1, batch2) $\to$ Tensor
`addbmm()`的原地运算形式
#### addcdiv(value=1, tensor1, tensor2) $\to$ Tensor
请查看`torch.addcdiv()`
#### addcdiv_(value=1, tensor1, tensor2) $\to$ Tensor
`addcdiv()`的原地运算形式
#### addcmul(value=1, tensor1, tensor2) $\to$ Tensor
请查看`torch.addcmul()`
#### addcmul_(value=1, tensor1, tensor2) $\to$ Tensor
`addcmul()`的原地运算形式
#### addmm(beta=1, mat, alpha=1, mat1, mat2) $\to$ Tensor
请查看`torch.addmm()`
#### addmm_(beta=1, mat, alpha=1, mat1, mat2) $\to$ Tensor
`addmm()`的原地运算形式
#### addmv(beta=1, tensor, alpha=1, mat, vec) $\to$ Tensor
请查看`torch.addmv()`
#### addmv_(beta=1, tensor, alpha=1, mat, vec) $\to$ Tensor
`addmv()`的原地运算形式
#### addr(beta=1, alpha=1, vec1, vec2) $\to$ Tensor
请查看`torch.addr()`
#### addr_(beta=1, alpha=1, vec1, vec2) $\to$ Tensor
`addr()`的原地运算形式
#### apply(callable) $\to$ Tensor
将函数`callable`作用于tensor中每一个元素，并将每个元素用`callable`返回值替代。
>！注意：
该函数只能在CPU tensor中使用，并且不应该用在有较高性能要求的代码块。
#### asin() $\to$ Tensor
请查看`torch.asin()`
#### asin_() $\to$ Tensor
`asin()`的原地运算形式
#### atan() $\to$ Tensor
请查看`torch.atan()`
#### atan2() $\to$ Tensor
请查看`torch.atan2()`
#### atan2_() $\to$ Tensor
`atan2()`的原地运算形式
#### atan_() $\to$ Tensor
`atan()`的原地运算形式
#### baddbmm(beta=1, alpha=1, batch1, batch2) $\to$ Tensor
请查看`torch.baddbmm()`
#### baddbmm_(beta=1, alpha=1, batch1, batch2) $\to$ Tensor
`baddbmm()`的原地运算形式
#### bernoulli() $\to$ Tensor
请查看`torch.bernoulli()`
#### bernoulli_() $\to$ Tensor
`bernoulli()`的原地运算形式
#### bmm(batch2) $\to$ Tensor
请查看`torch.bmm()`
#### byte() $\to$ Tensor
将tensor改为byte类型
#### bmm(median=0, sigma=1, *, generator=None) $\to$ Tensor
将tensor中元素用柯西分布得到的数值填充：
$$
P(x)={\frac1 \pi} {\frac \sigma {(x-median)^2 + \sigma^2}}
$$
#### ceil() $\to$ Tensor
请查看`torch.ceil()`
#### ceil_() $\to$ Tensor
`ceil()`的原地运算形式
#### char() $\to$ Tensor
将tensor元素改为char类型
#### chunk(n_chunks, dim=0) $\to$ Tensor
将tensor分割为tensor元组，请查看`torch.chunk()`
#### clamp(min, max) $\to$ Tensor
请查看`torch.clamp()`
#### clamp_(min, max) $\to$ Tensor
`clamp()`的原地运算形式
#### clone() $\to$ Tensor
返回与原tensor有相同大小和数据类型的tensor
#### contiguous() $\to$ Tensor
返回一个内存连续的有相同数据的tensor，如果原tensor内存连续则返回原tensor
#### copy_(src, async=False) $\to$ Tensor
将`src`中的元素复制到tensor中并返回tensor。
两个tensor应该有相同数目的元素，可能是不同的数据类型或存储在不同的设备上。
>参数：
>* src(Tensor)-复制的源tensor
>* async(bool)-如果为True并且复制是在CPU和GPU之间进行的，则复制后的拷贝可能会与原始信息同步，对于其他类型的复制操作则该参数不会发生作用。

#### cos() $\to$ Tensor
请查看`torch.cos()`
#### cos_() $\to$ Tensor
`cos()`的原地运算形式
#### cosh() $\to$ Tensor
请查看`torch.cosh()`
#### cosh_() $\to$ Tensor
`cosh()`的原地运算形式
#### cpu() $\to$ Tensor
如果在CPU上没有该tensor形式，则会返回一个CPU的拷贝
#### cross(other, dim=-1) $\to$ Tensor
请查看`torch.cross()`
#### cuda(device=None, async=False)
返回CUDA内存一个对象的副本
如果对象已近存在与CUDA存储中并且在正确的设备上，则不会进行复制并返回原始对象。

>参数：
>* device(int)-目的GPU的id，默认为当前的设备。
>* async(bool)-如果为True并且资源在固定内存中，则复制的副本将会与原始数据异步。否则，该参数没有意义。
#### cumprod(dim) $\to$ Tensor
请查看`torch.cumprod()`
#### cumsum(dim) $\to$ Tensor
请查看`torch.cumsum()`
#### data_ptr() $\to$ int
返回tensor第一个元素的地址
#### diag(diagonal=0) $\to$ Tensor
请查看`torch.diag()`
#### dim() $\to$ int
返回tensor的维数
#### dist(other, p=2) $\to$ Tensor
请查看`torch.dist()`
#### div(value)
请查看`torch.div()`
#### div_(value)
`div()`的原地运算形式
#### dot(tensor2) $\to$ float
请查看`torch.dot()`
#### double()
将该tensor投射位double类型
#### eig(eigenvectors=False) -> (Tensor, Tensor)
请查看`torch.eig()`
#### element_size() $\to$ int
返回单个元素的字节大小。
例：
```python
>>> torch.FloatTensor().element_size()
4
>>> torch.ByteTensor().element_size()
1
```
#### eq(other) $\to$ Tensor
请查看`torch.eq()`
#### eq_(other) $\to$ Tensor
`eq()`的原地运算形式
#### equal(other) $\to$ bool
请查看`torch.equal()`
#### exp() $\to$ Tensor
请查看`torch.exp()`
#### exp_() $\to$ Tensor
`exp()`的原地运算形式
#### expand(*sizes)
返回tensor的一个新视图，单个维度扩大为更大的尺寸。
tensor也可以扩大为更高维，新增加的维度将附在前面。
扩大tensor不需要分配新内存，只是仅仅新建一个tensor的视图，其中通过将`stride`设为0，一维将会扩展位更高维。任何一个一维的在不分配新内存情况下将会扩展为任意的数值。
>参数： *sizes(torch.Size or int...)-需要的扩展尺寸
例：
```python
>>> x = torch.Tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> x.expand(3, 4)
 1 1
 1 1
 2 2 2 2
 3 3 3 3
 [torch.FloatTensor of size 3x4]
```
#### expand_as(tensor)
将tensor扩展为确定大小的tensor。
该操作等效与：
```python
self.expand(tensor.size())
```
#### exponential_(lambd=1, *, generator=None) $to$ Tensor
将该tensor用指数分布得到的元素填充：
$$
P(x)=  \lambda e^{- \lambda x}
$$
#### fill_(value) $\to$ Tensor
将该tensor用指定的数值填充
#### float()
将tensor投射为float类型
#### floor() $\to$ Tensor
请查看`torch.floor()`
#### floor_() $\to$ Tensor
`floor()`的原地运算形式
#### fmod(divisor) $\to$ Tensor
请查看`torch.fmod()`
#### fmod_(divisor) $\to$ Tensor
`fmod()`的原地运算形式
#### frac() $\to$ Tensor
请查看`torch.frac()`
#### frac_() $\to$ Tensor
`frac()`的原地运算形式
#### gather(dim, index) $\to$ Tensor
请查看`torch.gather()`
#### ge(other) $\to$ Tensor
请查看`torch.ge()`
#### ge_(other) $\to$ Tensor
`ge()`的原地运算形式
#### gels(A) $\to$ Tensor
请查看`torch.gels()`
#### geometric_(p, *, generator=None) $\to$ Tensor
将该tensor用指数分布得到的元素填充：
$$
P(X=k)=  (1-p)^{k-1}p
$$
#### geqrf() -> (Tensor, Tensor)
请查看`torch.geqrf()`
#### ger(vec2) $\to$ Tensor
请查看`torch.ger()`
#### gesv(A) $\to$ Tensor, Tensor
请查看`torch.gesv()`
#### gt(other) $\to$ Tensor
请查看`torch.gt()`
#### gt_(other) $\to$ Tensor
`gt()`的原地运算形式
#### half()
将tensor投射为半精度浮点类型
#### histc(bins=100, min=0, max=0) $\to$ Tensor
请查看`torch.histc()`将该tensor用指数分布得到的元素填充：
$$
P(X=k)=  (1-p)^{k-1}p
$$
#### index(m) $\to$ Tensor
用一个二进制的掩码或沿着一个给定的维度从tensor中选取元素。`tensor.index(m)`与`tensor[m]`完全相同。
>参数： m(int or Byte Tensor or slice)-用来选取元素的维度或掩码
#### index_add_(dim, index, tensor) $\to$ Tensor
按参数index中的指数确定的顺序，将参数tensor中的元素加到原来的tensor中。参数tensor的尺寸必须严格地与原tensor匹配，否则会发生错误。
>参数：
>* dim(int)-索引index所指向的维度
>* index(LongTensor)-需要从tensor中选取的指数
>* tensor(Tensor)-含有相加元素的tensor
例：
```python
>>> x = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
>>> t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> index = torch.LongTensor([0, 2, 1])
>>> x.index_add_(0, index, t)
>>> x
  2 3 4
  8 9 10
  5 6 7
[torch.FloatTensor of size 3x3]
```
#### index_copy_(dim, index, tensor) $\to$ Tensor
按参数index中的指数确定的顺序，将参数tensor中的元素复制到原来的tensor中。参数tensor的尺寸必须严格地与原tensor匹配，否则会发生错误。
>参数：
>* dim(int)-索引index所指向的维度
>* index(LongTensor)-需要从tensor中选取的指数
>* tensor(Tensor)-含有被复制元素的tensor
例：
```python
>>> x = torch.Tensor(3， 3)
>>> t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> index = torch.LongTensor([0, 2, 1])
>>> x.index_copy_(0, index, t)
>>> x
  1 2 3
  7 8 9
  4 5 6
[torch.FloatTensor of size 3x3]
```
#### index_fill_(dim, index, val) $\to$ Tensor
按参数index中的指数确定的顺序，将参原tensor用参数`val`值填充。
>参数：
>* dim(int)-索引index所指向的维度
>* index(LongTensor)-指数
>* tensor(Tensor)-填充的值
例：
```python
>>> x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> index = torch.LongTensor([0, 2])
>>> x.index_fill_(0, index, -1)
>>> x
  -1 2 -1
  -1 5 -1
  -1 8 -1
[torch.FloatTensor of size 3x3]
```
#### index_select(dim, index) $\to$ Tensor
请查看`torch.index_select()`
#### int()
将该tensor投射为int类型
#### inverse() #\to# Tensor
请查看`torch.inverse()`
#### is_contiguous() $\to$ bool
如果该tensor在内存中是连续的则返回True。
#### is_cuda
#### is_pinned()
如果该tensor在固定内内存中则返回True
#### is_set_to(tensor) $\to$ bool
如果此对象引用与Torch C API相同的THTensor对象作为给定的张量，则返回True。
#### is_signed()
#### kthvalue(k, dim=None) -> (Tensor, LongTensor)
请查看`torch.kthvalue()`
#### le(other) $\to$ Tensor
请查看`torch.le()`
#### le_(other) $\to$ Tensor
`le()`的原地运算形式
#### lerp(start, end, weight)
请查看`torch.lerp()`
#### lerp_(*start, end, weight*) $\to$ Tensor
`lerp()`的原地运算形式
#### log() $\to$ Tensor
请查看`torch.log()`
#### loglp() $\to$ Tensor
请查看`torch.loglp()`
#### loglp_() $\to$ Tensor
`loglp()`的原地运算形式
#### log_()$\to$ Tensor
`log()`的原地运算形式
#### log_normal_(*mwan=1, std=2, *, gegnerator=None*)
将该tenso用均值为$\mu$,标准差为$\sigma$的对数正态分布得到的元素填充。要注意`mean`和`stdv`是基本正态分布的均值和标准差，不是返回的分布：
$$
P(X)= \frac {1} {x \sigma \sqrt {2 \pi}}e^{- \frac {(lnx- \mu)^2} {2 \sigma^2}}
$$
#### long()
将tensor投射为long类型
#### lt(*other*) $\to$ Tensor
请查看`torch.lt()`
#### lt_(*other*) $\to$ Tensor
`lt()`的原地运算形式
#### map_(*tensor, callable*)
将`callable`作用于本tensor和参数tensor中的每一个元素，并将结果存放在本tenor中。`callable`应该有下列标志：
```pyhton
def callable(a, b) -> number
```
#### masked_copy_(*mask, source*)
将`mask`中值为1元素对应的`source`中位置的元素复制到本tensor中。`mask`应该有和本tensor相同数目的元素。`source`中元素的个数最少为`mask`中值为1的元素的个数。
>参数：
>* mask(*ByteTensor*)-二进制掩码
>* source(*Tensor*)-复制的源tensor

>注意：
>`mask`作用于`self`自身的tensor，而不是参数中的`source`。
#### masked_fill_(*mask, value*)
在`mask`值为1的位置处用`value`填充。`mask`的元素个数需和本tensor相同，但尺寸可以不同。
>参数：
>* mask(*ByteTensor*)-二进制掩码
>* value(*Tensor*)-用来填充的值
#### masked_select(*mask*) $\to$ Tensor
请查看`torch.masked_select()`
#### max(*dim=None*) -> *float or(Tensor, Tensor)*
请查看`torch.max()`
#### mean(*dim=None*) -> *float or(Tensor, Tensor)*
请查看`torch.mean()`
#### median(*dim=-1, value=None, indices=None*) -> *(Tensor, LongTensor)*
请查看`torch.median()`
#### min(*dim=None*) -> *float or(Tensor, Tensor)*
请查看`torch.min()`
#### mm(*mat2*) $\to$ Tensor
请查看`torch.mm()`
#### mode(*dim=-1, value=None, indices=None*) -> *(Tensor, LongTensor)*
请查看`torch.mode()`
#### mul(*value*) $\to$ Tensor
请查看`torch.mul()`
#### mul_(*value*)
`mul()`的原地运算形式
#### multinomial(*num_samples, replacement=False, *, generator=None*) $\to$ Tensor
请查看`torch.multinomial()`
#### mv(*vec*) $\to$ Tensor
请查看`torch.mv()`
#### narrow(*dimension, start, length*) $\to$ Te
返回一个本tensor经过缩小后的tensor。维度`dim`缩小范围是`start`到`start+length`。原tensor与返回的tensor共享相同的底层内存。
>参数：
>* dimension(*int*)-需要缩小的维度
>* start(*int*)-起始维度
>* length(*int*)-
```python
>>> x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> x.narrow(0, 0, 2)
 1  2  3
 4  5  6
[torch.FloatTensor of size 2x3]
>>> x.narrow(1, 1, 2)
 2  3
 5  6
 8  9
[torch.FloatTensor of size 3x2]
```
#### ndimension() $\to$ int
`dim()`的另一种表示。
#### ne(*other*) $\to$ Tensor
请查看`torch.ne()`
#### ne_(*other*) $\to$ Tensor
`ne()`的原地运算形式
#### neg() $\to$ Tensor
请查看`torch.neg()`
#### neg_() $\to$ Tensor
`neg()`的原地运算形式
#### nelement() $\to$ int
`numel()`的另一种表示
#### new(_*args, **kwargs_)
构建一个有相同数据类型的tensor
#### nonezero() $\to$ LongTensor
请查看`torch.nonezero()
#### norm(*p=2*) $\to$ float
请查看`torch.norm()
#### normal_(*mean=0, std=1, *, gengerator=None*)
将tensor用均值为`mean`和标准差为`std`的正态分布填充。
#### numel() $\to$ int
请查看`numel()`
#### numpy() $\to$ ndarray
将该tensor以NumPy的形式返回`ndarray`，两者共享相同的底层内存。原tensor改变后会相应的影响`ndarray`，反之也一样。
#### orgqr(*input2*) $\to$ Tensor
请查看`torch.orgqr()`
#### ormqr(*input2, input3, left=True, transpose=False*) $\to$ Tensor
请查看`torch.ormqr()`
#### permute(*dims*)
将tensor的维度换位。
>参数： *dims(*int..*)-换位顺序
例：
```python
>>> x = torch.randn(2, 3, 5)
>>> x.size()
torch.Size([2, 3, 5])
>>> x.permute(2, 0, 1).size()
torch.Size([5, 2, 3])
```
#### pin_memory()
如果原来没有在固定内存中，则将tensor复制到固定内存中。
#### potrf(*upper=True*) $\to$ Tensor
请查看`torch.potrf()`
#### potri(*upper=True*) $\to$ Tensor
请查看`torch.potri()`
#### potrs(*input2, upper=True*) $\to$ Tensor
请查看`torch.potrs()`
#### pow(*exponent*)
请查看`torch.pow()`
#### pow_()
`pow()`的原地运算形式
#### prod()) $\to$ float
请查看`torch.prod()`
#### pstrf(*upper=True, tol=-1*) -> (*Tensor, IntTensor*)
请查看`torch.pstrf()`
#### qr()-> (*Tensor, IntTensor*)
请查看`torch.qr()`
#### random_(*from=0, to=None, *, generator=None*)
将tensor用从在[from, to-1]上的正态分布或离散正态分布取样值进行填充。如果没有明确，则填充值仅由本tensor的数据类型限定。
#### reciprocal() $\to$ Tensor
请查看`torch.reciprocal()`
#### reciprocal_() $\to$ Tensor
`reciprocal()`的原地运算形式
#### remainder(*divisor*) $\to$ Tensor
请查看`torch.remainder()`
#### remainder_(*divisor*) $\to$ Tensor
`remainder()`的原地运算形式
#### renorm(*p, dim, maxnorm*) $\to$ Tensor
请查看`torch.renorm()`
#### renorm_(*p, dim, maxnorm*) $\to$ Tensor
`renorm()`的原地运算形式
#### repeat(_*sizes_)
沿着指定的维度重复tensor。
不同于`expand()`，本函数复制的是tensor中的数据。
>参数： *sizes(_torch.Size ot int..._)-沿着每一维重复的次数
例：
```python
>>> x = torch.Tensor([1, 2, 3])
>>> x.repeat(4, 2)
 1  2  3  1  2  3
 1  2  3  1  2  3
 1  2  3  1  2  3
 1  2  3  1  2  3
[torch.FloatTensor of size 4x6]
>>> x.repeat(4, 2, 1).size()
torch.Size([4, 2, 3])
```
#### resize_(_*sizes_)
将tensor的大小调整指定的大小。如果元素个数比当前的内存大小大，就将底层存储大小调整为与新元素数目一致的大小。如果元素个数比当前内存小，则底层存储不会被改变。原来tensor中被保存下来的元素将保持不变，但新内存将不会被初始化。
>参数： *sizes(_torch.Size or int..._)-需要调整的大小
例：
```python
>>> x = torch.Tensor([[1, 2], [3, 4], [5, 6]])
>>> x.resize_(2, 2)
>>> x
 1  2
 3  4
[torch.FloatTensor of size 2x2]
```
#### resize_as_(_tensor_)
将本tensor的大小调整为与参数中的tensor相同的大小。等效于：
```python
self.resize_(tensor.size())
```
#### round() $\to$ Tensor
请查看`torch.round()`
#### round_() $\to$ Tensor
`round()`的原地运算形式
#### rsqrt() $\to$ Tensor
请查看`torch.rsqrt()`
#### rsqrt_() $\to$ Tensor
`rsqrt()`的原地运算形式
#### scatter_(*input, dim, index, src*) $\to$ Tensor
将`src`中的所有值按照`index`确定的索引写入本tensor中。其中索引是根据给定的dimension，dim按照`gather()`描述的规则来确定。
注意，index的值必须是在_0_到_(self.size(dim)-1)_之间，
>参数：
>* input(_Tensor_)-源tensor
>* dim(_int_)-索引的轴向
>* index(LongTensor)-
>* src(_Tensor or float_)-
例：
```python
>>> x = torch.rand(2, 5)
>>> x

 0.4319  0.6500  0.4080  0.8760  0.2355
 0.2609  0.4711  0.8486  0.8573  0.1029
[torch.FloatTensor of size 2x5]

>>> torch.zeros(3, 5).scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)

 0.4319  0.4711  0.8486  0.8760  0.2355
 0.0000  0.6500  0.0000  0.8573  0.0000
 0.2609  0.0000  0.4080  0.0000  0.1029
[torch.FloatTensor of size 3x5]

>>> z = torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 1.23)
>>> z

 0.0000  0.0000  1.2300  0.0000
 0.0000  0.0000  0.0000  1.2300
[torch.FloatTensor of size 2x4]
```
#### select(*dim, index*) $\to$ Tensor or number
按照index中选定的维度将tensor切片。如果tensor是一维的，则返回一个数字。否则，返回给定维度已经被移除的tensor。
>参数：
>* dim(_int_)-切片的维度
>* index(_int_)-用来选取的索引

>注意：
>`select()`等效于切片。例如，`tensor.select(0, index)`等效于`tensor[index]`，`tensor.select(2, index)`等效于`tensor[:, :, index]`.
#### set(_source=None, storage_offset=0, size=None, stride=None_)
设置底层内存，大小和步长。如果`tensor`是一个tensor，则将会与本tensor共享底层内存并且有相同的大小和步长。改变一个tensor中的元素将会反映在另一个tensor。
如果`source`是一个`Storage`，则将设置底层内存，偏移量，大小和步长。
>参数：
>* source(_Tensor or Storage_)-用到的tensor或内存
>* storage_offset(_int_)-内存的偏移量
>* size(_torch.Size_)-需要的大小，默认为源tensor的大小。
>* stride(_tuple_)-需要的步长，默认为C连续的步长。
#### share_memory_()
将底层内存移到共享内存中。
如果底层内存已经在共享内存中是将不进行任何操作。在共享内存中的tensor不能调整大小。
#### short()
将tensor投射为short类型。
#### sigmoid() $\to$ Tensor
请查看`torch.sigmoid()`
#### sigmoid_() $\to$ Tensor
`sidmoid()`的原地运算形式
#### sign() $\to$ Tensor
请查看`torch.sign()`
#### sign_() $\to$ Tensor
`sign()`的原地运算形式
#### sin() $\to$ Tensor
请查看`torch.sin()`
#### sin_() $\to$ Tensor
`sin()`的原地运算形式
#### sinh() $\to$ Tensor
请查看`torch.sinh()`
#### sinh_() $\to$ Tensor
`sinh()`的原地运算形式
#### size() $\to$ torch.Size
返回tensor的大小。返回的值是`tuple`的子类。
例：
```python
>>> torch.Tensor(3, 4, 5).size()
torch.Size([3, 4, 5])
```
#### sort(_dim=None, descending=False_) -> (_Tensor, LongTensor_)
请查看`torhc.sort()`
#### split(_split_size, dim=0_)
将tensor分割成tensor数组。
请查看`torhc.split()`
#### sqrt() $\to$ Tensor
请查看`torch.sqrt()`
#### sqrt_() $\to$ Tensor
`sqrt()`的原地运算形式
#### squeeze(_dim=None_) $\to$ Tensor
请查看`torch.squeeze()`
#### squeeze_(_dim=None_) $\to$ Tensor
`squeeze()`的原地运算形式
#### std() $\to$ float
请查看`torch.std()`
#### storage() $\to$ torch.Storage
返回底层内存。
#### storage_offset() $\to$ int
以储存元素的个数的形式返回tensor在地城内存中的偏移量。
例：
```python
>>> x = torch.Tensor([1, 2, 3, 4, 5])
>>> x.storage_offset()
0
>>> x[3:].storage_offset()
3
```
#### _classmethod()_ storage_type()
#### stride() $\to$ Tensor
返回tesnor的步长。
#### sub(_value, other_) $\to$ Tensor
从tensor中抽取一个标量或tensor。如果`value`和`other`都是给定的，则在使用之前`other`的每一个元素都会被`value`缩放。
#### sub_(_x_) $\to$ Tensor
`sub()`的原地运算形式
#### sum(_dim=None_) $\to$ Tensor
请查看`torch.sum()`
#### svd(_some=True_) -> (Tensor, Tensor, Tensor)
请查看`torch.svd()`
#### symeig(_eigenvectors=False, upper=True) -> (Tensor, Tensor)
请查看`torch.symeig()`
#### t() $\to$ Tensor
请查看`torch.t()`
#### t() $\to$ Tensor
`t()`的原地运算形式
#### tan() $\to$ Tensor
请查看`torch.tan()`
#### tan_() $\to$ Tensor
`tan()`的原地运算形式
#### tanh() $\to$ Tensor
请查看`torch.tanh()`
#### tanh_() $\to$ Tensor
`tanh()`的原地运算形式
#### tolist()
返回一个tensor的嵌套列表表示。
#### topk(_k, dim=None, largest=True, sorted=True_) -> (Tensor, LongTensor)
请查看`torch.topk()`
#### trace() $\to$ float
请查看`torch.trace()`
#### transpose(_dim0, dim1_) $\to$ Tensor
请查看`torch.transpose()`
#### transpose(_dim0, dim1_) $\to$ Tensor
`transpose()`的原地运算形式
#### tril(_k=0_) $\to$ Tensor
请查看`torch.tril()`
#### tril_(_k=0_) $\to$ Tensor
`tril()`的原地运算形式
#### triu(_k=0_) $\to$ Tensor
请查看`torch.triu()`
#### triu(_k=0_) $\to$ Tensor
`triu()`的原地运算形式
#### trtrs(_A, upper=True, transpose=False, unitriangular=False_) -> (Tensor, Tensor)
请查看`torch.trtrs()`
#### trunc() $\to$ Tensor
请查看`torch.trunc()`
#### trunc() $\to$ Tensor
`trunc()`的原地运算形式
#### type(_new_type=None, async=False_)
将对象投为指定的类型。
如果已经是正确的类型，则不会进行复制并返回原对象。
>参数：
>* new_type(_type or string_)-需要的类型
>* async(_bool_)-如果为True，并且源地址在固定内存中，目的地址在GPU或者相反，则会相对于源主异步执行复制。否则，该参数不发挥作用。
#### type_as(_tesnor_)
将tensor投射为参数给定tensor类型并返回。
如果tensor已经是正确的类型则不会执行操作。等效于：
```python
self.type(tensor.type())
```
>参数： *tensor(Tensor):有所需要类型的tensor
#### unfold(_dim, size, step_) $\to$ Tensor
返回一个tensor，其中含有在`dim`维tianchong度上所有大小为`size`的分片。两个分片之间的步长为`step`。
如果_sizedim_是dim维度的原始大小，则在返回tensor中的维度dim大小是_(sizedim-size)/step+1_
维度大小的附加维度将附加在返回的tensor中。
>参数：
>* dim(_int_)-需要展开的维度
>* size(_int_)-每一个分片需要展开的大小
>* step(_int_)-相邻分片之间的步长
例：
```python
>>> x = torch.arange(1, 8)
>>> x

 1
 2
 3
 4
 5
 6
 7
[torch.FloatTensor of size 7]

>>> x.unfold(0, 2, 1)

 1  2
 2  3
 3  4
 4  5
 5  6
 6  7
[torch.FloatTensor of size 6x2]

>>> x.unfold(0, 2, 2)

 1  2
 3  4
 5  6
[torch.FloatTensor of size 3x2]
```
#### uniform_(_from=0, to=1_) $\to$ Tensor
将tensor用从均匀分布中抽样得到的值填充。
#### unsqueeze(_dim_)
请查看`torch.unsqueeze()`
#### unsqueeze_(_dim_) $\to$ Tensor
`unsqueeze()`的原地运算形式
#### var()
请查看`torch.var()`
#### view(_*args_) $\to$ Tensor
返回一个有相同数据但大小不同的tensor。
返回的tensor必须有与原tensor相同的数据和相同数目的元素，但可以有不同的大小。一个tensor必须是连续的`contiguous()`才能被查看。
例：
```python
>>> x = torch.randn(4, 4)
>>> x.size()
torch.Size([4, 4])
>>> y = x.view(16)
>>> y.size()
torch.Size([16])
>>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
>>> z.size()
torch.Size([2, 8])
```
#### view_as(_tensor_)
返回被视作与给定的tensor相同大小的原tensor。
等效于：
```python
self.view(tensor.size())
```
#### zero_()
用0填充该tensor。
xit
