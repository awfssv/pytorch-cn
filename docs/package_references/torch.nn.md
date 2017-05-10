# torch.nn #

**内容：**

- **参数** 
- **容器**
- **卷积层**
- **池化层**
- **非线性激活函数**
- **正则化层**
- **循环层**
- **线性函数层**
- **Dropout层**
- **损失函数**
- **视觉层**
- **使用多GPU**
- **参数操作**

##**参数**##
**torch.nn.Parametr**

nn中的参数属于Vriable(torch.autograd.Vriable)的子类.  

**参数类中包含的方法：**  
-- **data:**得到参数的值(tensor类型)  
-- **requires_grad:**表示参数是否需要梯度（bool类型），如果需要梯度设置为true，不需要设置为False，默认为True  

##容器##

###class torch.nn.module###  
所有神经网络模块的基础类，你需要继承这个这个类，修改成你需要的神经网络结构  

    import torch.nn as nn
    import torch.nn.functional as F
    class Model(nn.Module):
    def __init__(self):
    super(Model, self).__init__()
    self.conv1 = nn.Conv2d(1, 20, 5)
    self.conv2 = nn.Conv2d(20, 20, 5)
    
    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
以下是module这个类的方法
    
###add_model(name, module)###
添加子模块给当前模块，常用作添加神经网络的层  

    #example  
    class Net(nn.Module):
	    def __init__(self):
	    super(Net, self).__init__()
	    
	    self.conv = nn.Sequential()
	    self.conv.add_module('conv1', nn.Conv2d(1, 10, kernel_size=5))
	    self.conv.add_module('relu1', nn.ReLU())
	    self.conv.add_module('pool1', nn.MaxPool2d(kernel_size=2))
	    self.conv.add_module('drop1', nn.Dropout(1 - p_keep_conv))
	    self.conv.add_module('conv2', nn.Conv2d(10, 20, kernel_size=5))
	    self.conv.add_module('relu2', nn.ReLU())
	    self.conv.add_module('pool2', nn.MaxPool2d(kernel_size=2))
	    self.conv.add_module('drop2', nn.Dropout(1 - p_keep_conv))
 
   
###children()###
返回子木块的迭代器
    
###cpu(device_id=None)####
将模型所有的参数和缓冲区迁移到CPU

###cuda(device_id=None)####
将模型所有的参数和缓冲区迁移到GPU  
参数： device_id，可以指定使用的GPU

###double()###
将所有参数和缓冲区转换为double数据类型

###evel()###
将模型设置为验证模式(在测试时使用，可以加速计算)

###float()###
将所有参数和缓冲区转换为float数据类型

###forward(*input)###
定义模型的前向计算。在所有的torch.nn.module的子类中，都应重构这个方法。

###half()###
将所有参数和缓冲区转换为half数据类型

###load_state_dict(state_dict)###
将state_dict保存的模型的参数和参数的缓存值，载入到你的模型中去。  
**参数 state_dict**：包含参数和参数的缓存值  

    class _netD(nn.Module):
      def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
	    # input is (nc) x 64 x 64
	    nn.Conv2d(3, 64, 4, 2, 1, bias=False),
	    nn.LeakyReLU(0.2, inplace=True),
	    # state size. (ndf) x 32 x 32
	    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
	    nn.BatchNorm2d(128),
	    nn.LeakyReLU(0.2, inplace=True),
	    # state size. (ndf*2) x 16 x 16
	    nn.Conv2d(128, 256, 4, 2, 1, bias=False),
	    nn.BatchNorm2d(256),
	    nn.LeakyReLU(0.2, inplace=True),
	    # state size. (ndf*4) x 8 x 8
	    nn.Conv2d(256, 512, 4, 2, 1, bias=False),
	    nn.BatchNorm2d(512),
	    nn.LeakyReLU(0.2, inplace=True),
	    # state size. (ndf*8) x 4 x 4
	    nn.Conv2d(512, 1, 4, 1, 0, bias=False),
	    nn.Sigmoid()
	    )
	  def forward(self, input):
	    self.mian(input)
	    return output.view(-1, 1)

载入模型    

    net = _netD()    
    net = load_state_dict("your module name")

###modules()###
返回包含网络中所有模块的一个迭代器  
**example**  

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    >>>     print(idx, '->', m)
    0 -> Sequential (
       (0): Linear (2 -> 2)
      (1): Linear (2 -> 2)
    )
    1 -> Linear (2 -> 2)    

###name_children()###
返回直接子模块的迭代器，包括模块的名称以及模块本身  
**example**  
    
    >>> for name, module in model.named_children():
    >>> if name in ['conv4', 'conv5']:
    >>> print(module)
  
###name_modules(memo=None, prefix='')###
返回网络中所有模块的迭代器，迭代器可以同时产生模块的名称以及模块的值。   
**example**
   
    #重复的模块只返回一次。 在以下示例中，l将只返回一次。
    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    >>> print(idx, '->', m)
    0 -> ('', Sequential (
      (0): Linear (2 -> 2)
      (1): Linear (2 -> 2)
    ))
    1 -> ('0', Linear (2 -> 2))

###parameters(memo=None)###
返回一个迭代器，迭代器可以产生模块的参数。常常使用这个函数将模型参数传递给优化器。  
**example**
    
    >>> for param in model.parameters():
    >>> print(type(param.data), param.size())
    <class 'torch.FloatTensor'> (20L,)
    <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

###register\_backward\_hook(hook)###
在模型的反向传播的hook里面设置一个register  
每次后向计算梯度传播时，这个hook会被计算
**example**  

    x = Variable(torch.randn(5, 5), requires_grad=True)
    y = x + 2
    y.register_hook(lambda grad: grad * -1)
    y.sum().backward()
    print x.grad # is now filled with -1

运行结果：
    
    Variable containing:
    -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1
    [torch.FloatTensor of size 5x5]

###register\_buffer(name, tensor)###
向模块添加一个缓冲区
这通常用于设置一个不被视为模型参数的缓冲区。 例如，在BatchNorm的running_mean不是模型的参数，但却是一直存在的状态。  
**example**  

    >>> self.register_buffer('running_mean', torch.zeros(num_features))

###register\_forward\_hook(hook)###
在模型的前向传播的hook里面设置一个register  
每次前向计算梯度传播时，这个hook会被计算 

###register_parameter(name, param)###
向模型中添加一个参数  
该参数可以使用给定的名称作为属性访问。

###state_dict(destination=None, prefix='')###
返回模块（字典数据类型）的关键字。  
包括参数和持续缓冲区（例如运行平均值）。关键字是相应的参数和缓冲区名称。  
**example**

    >>> module.state_dict().keys()
    ['bias', 'weight']

###train(mode=True)###
将模型设置为训练模式。这只对包含有诸如Dropout或BatchNorm的模型有效。

###zero_grad()###
将所有的梯度置为0

###class torch.nn.Sequential(*args)###
一个序列容器，模型可以有序的添加进这个容器里面。除此之外，还可以使用ordered字典，添加模型  
**example**

    # Example of using Sequential
    model = nn.Sequential(
      nn.Conv2d(1,20,5),
      nn.ReLU(),
      nn.Conv2d(20,64,5),
      nn.ReLU()
    )
    
    # Example of using Sequential with OrderedDict
    model = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(1,20,5)),
      ('relu1', nn.ReLU()),
      ('conv2', nn.Conv2d(20,64,5)),
      ('relu2', nn.ReLU())
    ]))
    
###class torch.nn.ModuleList(modules=None)###
在列表中保存子模块。  
ModuleList可以像普通的Python列表一样进行索引，将正确添加需要添加的模块，并且这些模块可以被Module的方法调用。  
**example**  

    class MyModule(nn.Module):
	    def __init__(self):
	    super(MyModule, self).__init__()
	    self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
	    
	    def forward(self, x):
	    # ModuleList can act as an iterable, or be indexed using ints
	    for i, l in enumerate(self.linears):
	    x = self.linears[i // 2](x) + l(x)
	    return x
 
###append(module)###
在模型列表的最后，添加一个给定的模型  
**Parameters:**module(nn.Module) --添加一个模型

###extend(modules)###
在模型列表的最后，添加多个给定的模型  
**Parameters：**	modules (list) –  添加多个给定的模型  

###class torch.nn.ParameterList(parameters=None)###
在列表中保存子模块。  
ModuleList可以像普通的Python列表一样进行索引，将正确添加需要添加的模块，并且这些模块可以被Module的方法调用

###append(parameter)###
在模型列表的最后，添加一个给定的参数  
**Parametes:**module(list) --添加一个参数

###extend(parameters)###
在模型列表的最后，添加一些给定的参数  
**Parameters:**module(list) --添加一些参数

#卷积层##
###class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)###
一维卷积层  
输入的尺度是($N, C_{in},L$)，输出尺度（$N,C_{out},L_{out}$）的计算方式：  
$$out(N_i, C_{out_j})=bias(C_{out_j})+\sum^{C_{in}-1}_{k=0}weight(C_{out_j},k)\bigotimes input(N_i,k)$$
**说明**
$\bigotimes$表示相关系数计算
stride控制相关系数的计算步长   
dilation用于控制内核点之间的距离，详细描述在这[此处输入链接的描述][1]
groups控制输入和输出之间的连接： group=1，输出是所有的输入的卷积；group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
**Parameters：**  

 1. in_channels (int) – 输入信号的通道
 2. out_channels (int) – 卷积产生的通道
 3. kerner\_size(int or tuple) - 卷积核的尺寸
 4. stride (int or tuple, optional) - 卷积步长
 5. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 6. dilation (int or tuple, optional) – 卷积核元素之间的间距
 7. groups (int, optional) – 从输入通道到输出通道的阻塞连接数
 8. bias (bool, optional) - 如果bias=True，添加偏置

**shape:**
input: $(N,C_{in},L_{in})$  
output: $(N,C_{out},L_{out})$
$L_{out}=floor((L_{in}+2*padding-dilation*(kernerl\_size-1)-1)/stride+1)$
**变量:**  
weight(Tensor) - 卷积的权重，shape是(out_channels, in_channels,kernel_size)  
bias(Tensor) - 卷积的偏置系数，shape是（out_channel）  
**example:**

    >>> m = nn.Conv1d(16, 33, 3, stride=2)
    >>> input = autograd.Variable(torch.randn(20, 16, 50))
    >>> output = m(input)
    
###class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)###
二维卷积层  
输入的尺度是($N, C_{in},H,W$)，输出尺度（$N,C_{out},H_{out},W_{out}$）的计算方式：  
$$out(N_i, C_{out_j})=bias(C_{out_j})+\sum^{C_{in}-1}_{k=0}weight(C_{out_j},k)\bigotimes input(N_i,k)$$
**说明**
$\bigotimes$表示二维的相关系数计算
stride控制相关系数的计算步长   
dilation用于控制内核点之间的距离，详细描述在这[此处输入链接的描述][2]
groups控制输入和输出之间的连接： group=1，输出是所有的输入的卷积；group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
参数kernel_size，stride,padding，dilation也可以是一个int的数据，此时卷积height和width值相同;也可以是一个tuple数组，tuple的第一维度表示height的数值，tuple的第二维度表示width的数值

**Parameters：**  

 1. in_channels (int) – 输入信号的通道
 2. out_channels (int) – 卷积产生的通道
 3. kerner\_size(int or tuple) - 卷积核的尺寸
 4. stride (int or tuple, optional) - 卷积步长
 5. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 6. dilation (int or tuple, optional) – 卷积核元素之间的间距
 7. groups (int, optional) – 从输入通道到输出通道的阻塞连接数
 8. bias (bool, optional) - 如果bias=True，添加偏置

**shape:**
input: $(N,C_{in},H_{in},W_{in})$  
output: $(N,C_{out},H_{out},W_{out})$
$H_{out}=floor((H_{in}+2*padding[0]-dilation[0]*(kernerl\_size[0]-1)-1)/stride[0]+1)$  
$W_{out}=floor((W_{in}+2*padding[1]-dilation[1]*(kernerl\_size[1]-1)-1)/stride[1]+1)$
**变量:**  
weight(Tensor) - 卷积的权重，shape是(out_channels, in_channels,kernel_size)  
bias(Tensor) - 卷积的偏置系数，shape是（out_channel）  
**example:**

    >>> # With square kernels and equal stride
    >>> m = nn.Conv2d(16, 33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    >>> # non-square kernels and unequal stride and with padding and dilation
    >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
    >>> output = m(input)

###class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)###
三维卷积层  
输入的尺度是($N, C_{in},D,H,W$)，输出尺度（$N,C_{out},D_{out},H_{out},W_{out}$）的计算方式：  
$$out(N_i, C_{out_j})=bias(C_{out_j})+\sum^{C_{in}-1}_{k=0}weight(C_{out_j},k)\bigotimes input(N_i,k)$$
**说明**
$\bigotimes$表示二维的相关系数计算
stride控制相关系数的计算步长   
dilation用于控制内核点之间的距离，详细描述在这[此处输入链接的描述][3]
groups控制输入和输出之间的连接： group=1，输出是所有的输入的卷积；group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
参数kernel_size，stride,padding，dilation也可以是一个int的数据 - 卷积height和width值相同，也可以是一个有三个int数据的tuple数组，tuple的第一维度表示depth的数值，tuple的第二维度表示height的数值，tuple的第三维度表示width的数值

**Parameters：**  

 1. in_channels (int) – 输入信号的通道
 2. out_channels (int) – 卷积产生的通道
 3. kerner\_size(int or tuple) - 卷积核的尺寸
 4. stride (int or tuple, optional) - 卷积步长
 5. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 6. dilation (int or tuple, optional) – 卷积核元素之间的间距
 7. groups (int, optional) – 从输入通道到输出通道的阻塞连接数
 8. bias (bool, optional) - 如果bias=True，添加偏置

**shape:**
input: $(N,C_{in},D_{in},H_{in},W_{in})$  
output: $(N,C_{out},D_{out},H_{out},W_{out})$  
$D_{out}=floor((D_{in}+2*padding[0]-dilation[0]*(kernerl\_size[0]-1)-1)/stride[0]+1)$ 
$H_{out}=floor((H_{in}+2*padding[1]-dilation[2]*(kernerl\_size[1]-1)-1)/stride[1]+1)$  
$W_{out}=floor((W_{in}+2*padding[2]-dilation[2]*(kernerl\_size[2]-1)-1)/stride[2]+1)$
**变量:**  
weight(Tensor) - 卷积的权重，shape是(out_channels, in_channels,kernel_size)  
bias(Tensor) - 卷积的偏置系数，shape是（out_channel）  
**example:**

    >>> # With square kernels and equal stride
    >>> m = nn.Conv3d(16, 33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
    >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
    >>> output = m(input)
    
###class torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)###
1维的解卷积操作（transposed convolution operator，注意改视作操作可视作解卷积操作，但并不是真正的解卷积操作）
该模块可以看作是Conv1d相对于其输入的梯度，有时（但不正确地）被称为解卷积操作。

**注意**
由于内核的大小，输入的最后的一些列的数据可能会丢失。因为输入和输出是不是完全的互相关。因此，用户可以进行适当的填充（padding操作）。  

**参数**

 1. in_channels (int) – 输入信号的通道数
 2. out_channels (int) – 卷积产生的通道数
 3. kerner\_size(int or tuple) - 卷积核的大小
 4. stride (int or tuple, optional) - 卷积步长
 5. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 6. utput_padding (int or tuple, optional) - 输出的每一条边补充0的层数
 6. dilation (int or tuple, optional) – 卷积核元素之间的间距
 7. groups (int, optional) – 从输入通道到输出通道的阻塞连接数
 8. bias (bool, optional) - 如果bias=True，添加偏置

**shape:**
input: $(N,C_{in},L_{in})$  
output: $(N,C_{out},L_{out})$  
$L_{out}=(L_{in}-1)*stride-2*padding+kernel\_size+output\_padding$

**变量:**  
`weight(Tensor)` - 卷积的权重，大小是(in_channels, in_channels,kernel_size)  
`bias(Tensor)` - 卷积的偏置系数，大小是（out_channel） 

###class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
2维的转置卷积操作（transposed convolution operator，注意改视作操作可视作解卷积操作，但并不是真正的解卷积操作）
该模块可以看作是Conv2d相对于其输入的梯度，有时（但不正确地）被称为解卷积操作。

**说明**
 
stride控制相关系数的计算步长   
dilation用于控制内核点之间的距离，详细描述在这[此处输入链接的描述][4]
groups控制输入和输出之间的连接： group=1，输出是所有的输入的卷积；group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
参数kernel_size，stride,padding，dilation数据类型：
一个int类型的数据，此时卷积height和width值相同;
也可以是一个tuple数组（包含来两个int类型的数据），第一个int数据表示height的数值，tuple的第二个int类型的数据表示width的数值

**注意**
由于内核的大小，输入的最后的一些列的数据可能会丢失。因为输入和输出是不是完全的互相关。因此，用户可以进行适当的填充（padding操作）。 

**参数：**

 1. in_channels (int) – 输入信号的通道数
 2. out_channels (int) – 卷积产生的通道数
 3. kerner\_size(int or tuple) - 卷积核的大小
 4. stride (int or tuple, optional) - 卷积步长
 5. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 6. utput_padding (int or tuple, optional) - 输出的每一条边补充0的层数
 6. dilation (int or tuple, optional) – 卷积核元素之间的间距
 7. groups (int, optional) – 从输入通道到输出通道的阻塞连接数
 8. bias (bool, optional) - 如果bias=True，添加偏置

**shape:**
input: $(N,C_{in},H_{in}，W_{in})$  
output: $(N,C_{out},H_{out},W_{out})$  
$H_{out}=(H_{in}-1)*stride[0]-2*padding[0]+kernel\_size[0]+output\_padding[0]$
$W_{out}=(W_{in}-1)*stride[1]-2*padding[1]+kernel\_size[1]+output\_padding[1]$

**变量:**  
`weight(Tensor)` - 卷积的权重，大小是(in_channels, in_channels,kernel_size)  
`bias(Tensor)` - 卷积的偏置系数，大小是（out_channel） 

**Example**

    >>> # With square kernels and equal stride
    >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
    >>> output = m(input)
    >>> # exact output size can be also specified as an argument
    >>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
    >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
    >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
    >>> h = downsample(input)
    >>> h.size()
    torch.Size([1, 16, 6, 6])
    >>> output = upsample(h, output_size=input.size())
    >>> output.size()
    torch.Size([1, 16, 12, 12])

###class torch.nn.ConvTranspose3d(in\_channels, out\_channels, kernel\_size, stride=1, padding=0, output\_padding=0, groups=1, bias=True)
3维的转置卷积操作（transposed convolution operator，注意改视作操作可视作解卷积操作，但并不是真正的解卷积操作）
转置卷积操作将每个输入值和一个可学习权重的卷积核相乘，输出所有输入通道的求和

该模块可以看作是Conv3d相对于其输入的梯度，有时（但不正确地）被称为解卷积操作。

**说明**
 
stride控制相关系数的计算步长   
dilation用于控制内核点之间的距离，详细描述在这[此处输入链接的描述][5]
groups控制输入和输出之间的连接： group=1，输出是所有的输入的卷积；group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
参数kernel\_size，stride, padding，dilation数据类型：
一个int类型的数据，此时卷积height和width值相同;
也可以是一个tuple数组（包含来两个int类型的数据），第一个int数据表示height的数值，tuple的第二个int类型的数据表示width的数值

**注意**
由于内核的大小，输入的最后的一些列的数据可能会丢失。因为输入和输出是不是完全的互相关。因此，用户可以进行适当的填充（padding操作）。 

**参数：**

 1. in\_channels (int) – 输入信号的通道数
 2. out\_channels (int) – 卷积产生的通道数
 3. kerner\_size(int or tuple) - 卷积核的大小
 4. stride (int or tuple, optional) - 卷积步长
 5. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 6. output\_padding (int or tuple, optional) - 输出的每一条边补充0的层数
 6. dilation (int or tuple, optional) – 卷积核元素之间的间距
 7. groups (int, optional) – 从输入通道到输出通道的阻塞连接数
 8. bias (bool, optional) - 如果bias=True，添加偏置

**shape:**
input: $(N,C_{in},H_{in}，W_{in})$  
output: $(N,C_{out},H_{out},W_{out})$ 
$D_{out}=(D_{in}-1)*stride[0]-2*padding[0]+kernel\_size[0]+output\_padding[0]$
$H_{out}=(H_{in}-1)*stride[1]-2*padding[1]+kernel\_size[1]+output\_padding[0]$
$W_{out}=(W_{in}-1)*stride[2]-2*padding[2]+kernel\_size[2]+output\_padding[2]$

**变量:**  
`weight(Tensor)` - 卷积的权重，大小是(in\_channels, in\_channels,kernel\_size)  
`bias(Tensor)` - 卷积的偏置系数，大小是（out\_channel） 

**Example**

    >>> # With square kernels and equal stride
    >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
    >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
    >>> output = m(input)

##池化层##

###class torch.nn.MaxPool1d(kernel\_size, stride=None, padding=0, dilation=1, return\_indices=False, ceil_mode=False)

对于输入信号的输入通道，提供1维最大池化（max pooling）操作

如果输入的大小是$(N,C,L)$，那么输出的大小是$(N,C,L_{out})$的计算方式是：
$out(N_i, C_j,k)=max^{kernel\_size-1}_{m=0}input(N_{i},C_j,stride*k+m)$

如果padding不是0，会在输入的每一边添加相应数目0
dilation用于控制内核点之间的距离，详细描述在这[此处输入链接的描述][6]

**参数：**

 1. kerner\_size(int or tuple) - max pooling的窗口大小
 2. stride (int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
 3. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 4. dilation (int or tuple, optional) – 一个控制窗口中元素步幅的参数
 5. return_indices - 如果等于true，会返回输出最大值的序号，对于上采样操作会有帮助
 6. ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

**shape:**
input: $(N,C_{in},L_{in})$  
output: $(N,C_{out},L_{out})$  
$L_{out}=floor((L_{in} + 2*padding - dilation*(kernel\_size - 1) - 1)/stride + 1$ 

**example:**

    >>> # pool of size=3, stride=2
    >>> m = nn.MaxPool1d(3, stride=2)
    >>> input = autograd.Variable(torch.randn(20, 16, 50))
    >>> output = m(input)

###class torch.nn.MaxPool1d(kernel\_size, stride=None, padding=0, dilation=1, return\_indices=False, ceil\_mode=False)

对于输入信号的输入通道，提供2维最大池化（max pooling）操作

如果输入的大小是$(N,C,H,W)$，那么输出的大小是$(N,C,H_{out},W_{out})$和池化窗口大小$(kH,kW)$的关系是：
$out(N_i, C_j,k)=max^{kH-1}_{m=0}max^{kW-1}_{m=0}input(N_{i},C_j,stride[0]*h+m,stride[1]*w+n)$

如果padding不是0，会在输入的每一边添加相应数目0
dilation用于控制内核点之间的距离，详细描述在这[此处输入链接的描述][7]

参数kernel\_size，stride, padding，dilation数据类型：
一个int类型的数据，此时卷积height和width值相同;
也可以是一个tuple数组（包含来两个int类型的数据），第一个int数据表示height的数值，tuple的第二个int类型的数据表示width的数值

**参数：**

 1. kerner\_size(int or tuple) - max pooling的窗口大小
 2. stride (int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
 3. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 4. dilation (int or tuple, optional) – 一个控制窗口中元素步幅的参数
 5. return_indices - 如果等于true，会返回输出最大值的序号，对于上采样操作会有帮助
 6. ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

**shape:**
input: $(N,C,H_{in},W_{in})$  
output: $(N,C,H_{out},W_{out})$  
$H_{out}=floor((H_{in} + 2*padding[0] - dilation[0]*(kernel\_size[0] - 1) - 1)/stride[0] + 1$ 
$W_{out}=floor((W_{in} + 2*padding[1] - dilation[1]*(kernel\_size[1] - 1) - 1)/stride[1] + 1$ 

**example:**

    >>> # pool of square window of size=3, stride=2
    >>> m = nn.MaxPool2d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
    >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
    >>> output = m(input)

###class torch.nn.MaxPool3d(kernel\_size, stride=None, padding=0, dilation=1, return\_indices=False, ceil\_mode=False)

对于输入信号的输入通道，提供3维最大池化（max pooling）操作

如果输入的大小是$(N,C,D,H,W)$，那么输出的大小是$(N,C,D,H_{out},W_{out})$和池化窗口大小$(kD,kH,kW)$的关系是：
$out(N_i, C_j,d,h,w)=max^{kD-1}_{m=0}max^{kH-1}_{m=0}max^{kW-1}_{m=0}input(N_{i},C_j,stride[0]*k+d,stride[1]*h+m,stride[2]*w+n)$

如果padding不是0，会在输入的每一边添加相应数目0
dilation用于控制内核点之间的距离，详细描述在这[此处输入链接的描述][8]

参数kernel\_size，stride, padding，dilation数据类型：
一个int类型的数据，此时卷积height和width值相同;
也可以是一个tuple数组（包含来两个int类型的数据），第一个int数据表示height的数值，tuple的第二个int类型的数据表示width的数值

**参数：**

 1. kerner\_size(int or tuple) - max pooling的窗口大小
 2. stride (int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
 3. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 4. dilation (int or tuple, optional) – 一个控制窗口中元素步幅的参数
 5. return_indices - 如果等于true，会返回输出最大值的序号，对于上采样操作会有帮助
 6. ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

**shape:**
input: $(N,C,H_{in},W_{in})$  
output: $(N,C,H_{out},W_{out})$  
$D_{out}=floor((D_{in} + 2*padding[0] - dilation[0]*(kernel\_size[0] - 1) - 1)/stride[0] + 1)$ 
$H_{out}=floor((H_{in} + 2*padding[1] - dilation[1]*(kernel\_size[0] - 1) - 1)/stride[1] + 1)$ 
$W_{out}=floor((W_{in} + 2*padding[2] - dilation[2]*(kernel\_size[2] - 1) - 1)/stride[2] + 1)$ 

**example:**

    >>> # pool of square window of size=3, stride=2
    >>> m = nn.MaxPool3d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
    >>> input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
    >>> output = m(input)


####class torch.nn.MaxUnpool1d(kernel\_size, stride=None, padding=0)
Maxpool1d的逆过程，不过并不是完全的逆过程，因为在maxpool1d的过程中，一些最大值的已经丢失。
MaxUnpool1d输入MaxPool1d的输出，包括最大值的索引，并计算所有maxpool1d过程中非最大值被设置为零的部分的反向。

**注意：**
MaxPool1d可以将多个输入大小映射到相同的输出大小。因此，反演过程可能会变得模棱两可。 为了适应这一点，可以在调用中将输出大小（output_size）作为额外的参数传入。 具体用法，请参阅下面的输入和示例

**参数：**

 1. kerner\_size(int or tuple) - max pooling的窗口大小
 2. stride (int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
 3. padding (int or tuple, optional) - 输入的每一条边补充0的层数

**输入：**
input:需要转换的tensor
indices：Maxpool1d的索引号
output_size:一个指定输出大小的torch.Size

**shape:**
input: $(N,C,H_{in})$
output:$(N,C,H_{out})$
$H_{out}=(H_{in}-1)*stride[0]-2*padding[0]+kernel\_size[0]$
也可以使用output_size指定输出的大小

**Example：**

    >>> pool = nn.MaxPool1d(2, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool1d(2, stride=2)
    >>> input = Variable(torch.Tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]]))
    >>> output, indices = pool(input)
    >>> unpool(output, indices)
    Variable containing:
    (0 ,.,.) =
       0   2   0   4   0   6   0   8
    [torch.FloatTensor of size 1x1x8]
    
    >>> # Example showcasing the use of output_size
    >>> input = Variable(torch.Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]))
    >>> output, indices = pool(input)
    >>> unpool(output, indices, output_size=input.size())
    Variable containing:
    (0 ,.,.) =
       0   2   0   4   0   6   0   8   0
    [torch.FloatTensor of size 1x1x9]
    
    >>> unpool(output, indices)
    Variable containing:
    (0 ,.,.) =
       0   2   0   4   0   6   0   8
    [torch.FloatTensor of size 1x1x8]


####class torch.nn.MaxUnpool2d(kernel\_size, stride=None, padding=0)
Maxpool2d的逆过程，不过并不是完全的逆过程，因为在maxpool1d的过程中，一些最大值的已经丢失。
MaxUnpool2d输入MaxPool1d的输出，包括最大值的索引，并计算所有maxpool2d过程中非最大值被设置为零的部分的反向。

**注意：**
MaxPool2d可以将多个输入大小映射到相同的输出大小。因此，反演过程可能会变得模棱两可。 为了适应这一点，可以在调用中将输出大小（output_size）作为额外的参数传入。 具体用法，请参阅下面的输入和示例

**参数：**

 1. kerner\_size(int or tuple) - max pooling的窗口大小
 2. stride (int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
 3. padding (int or tuple, optional) - 输入的每一条边补充0的层数

**输入：**
input:需要转换的tensor
indices：Maxpool1d的索引号
output_size:一个指定输出大小的torch.Size

**shape:**
input: $(N,C,H_{in},W_{in})$
output:$(N,C,H_{out},W_{out})$
$H_{out}=(H_{in}-1)*stride[0]-2*padding[0]+kernel\_size[0]$
$W_{out}=(W_{in}-1)*stride[1]-2*padding[1]+kernel\_size[1]$
也可以使用output_size指定输出的大小

**Example：**

    >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool2d(2, stride=2)
    >>> input = Variable(torch.Tensor([[[[ 1,  2,  3,  4],
    ...                                  [ 5,  6,  7,  8],
    ...                                  [ 9, 10, 11, 12],
    ...                                  [13, 14, 15, 16]]]]))
    >>> output, indices = pool(input)
    >>> unpool(output, indices)
    Variable containing:
    (0 ,0 ,.,.) =
       0   0   0   0
       0   6   0   8
       0   0   0   0
       0  14   0  16
    [torch.FloatTensor of size 1x1x4x4]
    
    >>> # specify a different output size than input size
    >>> unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))
    Variable containing:
    (0 ,0 ,.,.) =
       0   0   0   0   0
       6   0   8   0   0
       0   0   0  14   0
      16   0   0   0   0
       0   0   0   0   0
    [torch.FloatTensor of size 1x1x5x5]

####class torch.nn.MaxUnpool3d(kernel\_size, stride=None, padding=0)
Maxpool2d的逆过程，不过并不是完全的逆过程，因为在maxpool1d的过程中，一些最大值的已经丢失。
MaxUnpool2d输入MaxPool1d的输出，包括最大值的索引，并计算所有maxpool2d过程中非最大值被设置为零的部分的反向。

**注意：**
MaxPool2d可以将多个输入大小映射到相同的输出大小。因此，反演过程可能会变得模棱两可。为了适应这一点，可以在调用中将输出大小（output_size）作为额外的参数传入。具体用法，请参阅下面的输入和示例

**参数：**

 1. kerner\_size(int or tuple) - Maxpooling窗口大小
 2. stride (int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
 3. padding (int or tuple, optional) - 输入的每一条边补充0的层数

**输入：**
input:需要转换的tensor
indices：Maxpool1d的索引号
output_size:一个指定输出大小的torch.Size

**shape:**
input: $(N,C,D_{in},H_{in},W_{in})$
output:$(N,C,D_{out},H_{out},W_{out})$
$D_{out}=(D_{in}-1)*stride[0]-2*padding[0]+kernel\_size[0]$
$H_{out}=(H_{in}-1)*stride[1]-2*padding[0]+kernel\_size[1]$
$W_{out}=(W_{in}-1)*stride[2]-2*padding[2]+kernel\_size[2]$
也可以使用output_size指定输出的大小

**Example：**

    >>> # pool of square window of size=3, stride=2
    >>> pool = nn.MaxPool3d(3, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool3d(3, stride=2)
    >>> output, indices = pool(Variable(torch.randn(20, 16, 51, 33, 15)))
    >>> unpooled_output = unpool(output, indices)
    >>> unpooled_output.size()
    torch.Size([20, 16, 51, 33, 15])

###class torch.nn.AvgPool1d(kernel\_size, stride=None, padding=0, ceil\_mode=False, count\_include_pad=True)

对信号的输入通道，提供1维平均池化（average pooling ）
输入信号的大小$(N,C,L)$，输出大小$(N,C,L_{out})$和池化窗口大小$k$的关系是：    
$out(N_i,C_j,l)=1/k*\sum^{k}_{m=0}input(N_{i},C_{j},stride*l+m)$
如果padding不是0，会在输入的每一边添加相应数目0

**参数：**

 1. kerner\_size(int or tuple) - 池化窗口大小
 2. stride (int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
 3. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 4. dilation (int or tuple, optional) – 一个控制窗口中元素步幅的参数
 5. return_indices - 如果等于true，会返回输出最大值的序号，对于上采样操作会有帮助
 6. ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

**shape：**
input:$(N,C,L_{in})$
output:$(N,C,L_{out})$
$L_{out}=floor((L_{in}+2*padding-kernel\_size)/stride+1)$

**Example:**

    >>> # pool with window of size=3, stride=2
    >>> m = nn.AvgPool1d(3, stride=2)
    >>> m(Variable(torch.Tensor([[[1,2,3,4,5,6,7]]])))
    Variable containing:
    (0 ,.,.) =
      2  4  6
    [torch.FloatTensor of size 1x1x3]

###class torch.nn.AvgPool2d(kernel\_size, stride=None, padding=0, ceil\_mode=False, count\_include_pad=True)

对信号的输入通道，提供2维的平均池化（average pooling ）
输入信号的大小$(N,C,H,W)$，输出大小$(N,C,H_{out},W_{out})$和池化窗口大小$(kH,kW)$的关系是：    
$out(N_i,C_j,h,w)=1/(kH*kW)*\sum^{kH-1}_{m=0}\sum^{kW-1}_{n=0}input(N_{i},C_{j},stride[0]*h+m,stride[1]*w+n)$
如果padding不是0，会在输入的每一边添加相应数目0

**参数：**

 1. kerner\_size(int or tuple) - 池化窗口大小
 2. stride (int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
 3. padding (int or tuple, optional) - 输入的每一条边补充0的层数
 4. dilation (int or tuple, optional) – 一个控制窗口中元素步幅的参数
 5. ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作
 6. count_include_pad - 如果等于True，计算平均池化时，将包括padding填充的0

**shape：**
input:$(N,C,H_{in},W_{in})$
output:$(N,C,H_{out},W_{out})$
$H_{out}=floor((H_{in}+2*padding[0]-kernel\_size[0])/stride[0]+1)$
$W_{out}=floor((W_{in}+2*padding[1]-kernel\_size[1])/stride[1]+1)$

**Example:**

    >>> # pool of square window of size=3, stride=2
    >>> m = nn.AvgPool2d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
    >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
    >>> output = m(input)

###class torch.nn.AvgPool3d(kernel\_size, stride=None)

对信号的输入通道，提供3维的平均池化（average pooling）
输入信号的大小$(N,C,D,H,W)$，输出大小$(N,C,D_{out},H_{out},W_{out})$和池化窗口大小$(kD,kH,kW)$的关系是：    
$out(N_i,C_j,d,h,w)=1/(kD*kH*kW)*\sum^{kD-1}_{k=0}\sum^{kH-1}_{m=0}\sum^{kW-1}_{n=0}input(N_{i},C_{j},stride[0]*d+k,stride[1]*h+m,stride[2]*w+n)$
如果padding不是0，会在输入的每一边添加相应数目0

**参数：**

 1. kerner\_size(int or tuple) - 池化窗口大小
 2. stride (int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
 


**shape：**
输入大小:$(N,C,D_{in},H_{in},W_{in})$
输出大小:$(N,C,D_{out},H_{out},W_{out})$
$D_{out}=floor((D_{in}+2*padding[0]-kernel\_size[0])/stride[0]+1)$
$H_{out}=floor((H_{in}+2*padding[1]-kernel\_size[1])/stride[1]+1)$
$W_{out}=floor((W_{in}+2*padding[2]-kernel\_size[2])/stride[2]+1)$

**Example:**

    >>> # pool of square window of size=3, stride=2
    >>> m = nn.AvgPool3d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
    >>> input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
    >>> output = m(input)

###class torch.nn.FractionalMaxPool2d(kernel\_size, output\_size=None, output\_ratio=None, return\_indices=False, \_random\_samples=None)

对sh输入的信号，提供2维的分数最大化池化操作

分数最大化池化的细节请阅读论文[此处输入链接的描述][9]

由目标输出大小确定的随机步长,在$kH*kW$区域进行最大池化操作。 输出特征和输入特征的数量相同。

**参数：**

1. kerner\_size(int or tuple) - 最大池化操作时的窗口大小。可以是一个数字（表示K*K的窗口），也可以是一个元组（khxkw）
2. output\_size - 输出图像的尺寸。可以使用一个tuple指定(oH,oW)，也可以使用一个数字oH指定一个oH*oH的输出。
3. output_ratio – 将输入图像的大小的百分比指定为输出图片的大小，使用一个范围在(0,1)之间的数字指定   
4. return_indices - 默认值False，如果设置为True，会返回输出的索引。对  nn.MaxUnpool2d有用。

**Example：**

    >>> # pool of square window of size=3, and target output size 13x12
    >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
    >>> # pool of square window and target output size being half of input image size
    >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
    >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
    >>> output = m(input)

###class torch.nn.LPPool2d(norm\_type, kernel\_size, stride=None, ceil\_mode=False)
对输入信号提供2维的幂平均池化操作。
输出的计算方式：&f(x)=pow(sum(X,p),1/p)&

 - 当p=无限大时，等价于最大池化操作
 - 当p=1时，等价于平均池化操作

参数kernel_size, stride的数据类型：

 - int，池化窗口的宽和高相等
 - tuple数组（两个数字的），一个元素是池化窗口的高，另一个是宽

 **参数**
 

 - kernel\_size: 池化窗口的大小
 - stride：池化窗口移动的步长。kernel\_size是默认值
 - ceil\_mode: 当ceil\_mode等与True，将使用向下取整代替向上取整

**shape**

 - 输入：$(N,C,H_{in},W_{in})$
 - 输出：$(N,C,H_{out},W_{out})$
$$H_{out} = floor((H_{in}+2*padding[0]-dilation[0]*(kernel\_size[0]-1)-1)/stride[0]+1)$$
$$W_{out} = floor((W_{in}+2*padding[1]-dilation[1]*(kernel\_size[1]-1)-1)/stride[1]+1)$$ 
 
**Example:** 

    >>> # power-2 pool of square window of size=3, stride=2
    >>> m = nn.LPPool2d(2, 3, stride=2)
    >>> # pool of non-square window of power 1.2
    >>> m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
    >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
    >>> output = m(input)

###class torch.nn.AdaptiveMaxPool1d(output\_size, return\_indices=False)
对输入信号，提供1维的自适应最大池化操作
对于任何输入大小的输入，可以将输出尺寸指定为H，但是输入和输出特征的数目不会变化。

**参数：**

 - output\_size: 输出信号的尺寸
 - return\_indices: 如果设置为True，会返回输出的索引。对  nn.MaxUnpool1d有用，默认值是False

**Example：**

    >>> # target output size of 5
    >>> m = nn.AdaptiveMaxPool1d(5)
    >>> input = autograd.Variable(torch.randn(1, 64, 8))
    >>> output = m(input) 

###class torch.nn.AdaptiveMaxPool2d(output\_size, return\_indices=False)
对输入信号，提供2维的自适应最大池化操作
对于任何输入大小的输入，可以将输出尺寸指定为H*W，但是输入和输出特征的数目不会变化。

**参数：**

 - output\_size: 输出信号的尺寸,可以用（H,W）表示H*W的输出，也可以使用耽搁数字H表示H*H大小的输出
 - return\_indices: 如果设置为True，会返回输出的索引。对  nn.MaxUnpool2d有用，默认值是False

**Example：** 

    >>> # target output size of 5x7
    >>> m = nn.AdaptiveMaxPool2d((5,7))
    >>> input = autograd.Variable(torch.randn(1, 64, 8, 9))
    >>> # target output size of 7x7 (square)
    >>> m = nn.AdaptiveMaxPool2d(7)
    >>> input = autograd.Variable(torch.randn(1, 64, 10, 9))
    >>> output = m(input)

###class torch.nn.AdaptiveAvgPool1d(output\_size)
对输入信号，提供1维的自适应平均池化操作
对于任何输入大小的输入，可以将输出尺寸指定为H*W，但是输入和输出特征的数目不会变化。

**参数：**

 - output\_size: 输出信号的尺寸  

**Example：** 

    >>> # target output size of 5
    >>> m = nn.AdaptiveAvgPool1d(5)
    >>> input = autograd.Variable(torch.randn(1, 64, 8))
    >>> output = m(input)

###class torch.nn.AdaptiveAvgPool2d(output\_size)
对输入信号，提供2维的自适应平均池化操作
对于任何输入大小的输入，可以将输出尺寸指定为H*W，但是输入和输出特征的数目不会变化。

**参数：**

 - output\_size: 输出信号的尺寸,可以用（H,W）表示H*W的输出，也可以使用耽搁数字H表示H*H大小的输出

**Example：**

    >>> # target output size of 5x7
    >>> m = nn.AdaptiveAvgPool2d((5,7))
    >>> input = autograd.Variable(torch.randn(1, 64, 8, 9))
    >>> # target output size of 7x7 (square)
    >>> m = nn.AdaptiveAvgPool2d(7)
    >>> input = autograd.Variable(torch.randn(1, 64, 10, 9))
    >>> output = m(input)

 
 
 
 
 
 
 
 
 
 
 
  [1]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
  [2]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
  [3]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
  [4]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
  [5]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
  [6]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
  [7]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
  [8]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
  [9]: http://arxiv.org/abs/1412.6071