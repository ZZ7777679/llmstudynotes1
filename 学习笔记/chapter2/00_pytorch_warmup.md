# pytorch warmup学习笔记

- [pytorch warmup学习笔记](#pytorch-warmup学习笔记)
  - [张量维度变换](#张量维度变换)
    - [基本功：原生tensor维度变换操作复习与拓展](#基本功原生tensor维度变换操作复习与拓展)
    - [原生tensor维度变换的缺点与einops的优势](#原生tensor维度变换的缺点与einops的优势)
  - [嵌入层的本质](#嵌入层的本质)
  - [前向传播与反向传播](#前向传播与反向传播)
    - [前向传播](#前向传播)
    - [反向传播](#反向传播)
  
## 张量维度变换

### 基本功：原生tensor维度变换操作复习与拓展
  
  torch.reshape与torch.view都是改变张量维度

  + torch.view要求张量在内存中连续，否则报错；torch.reshape不要求连续，安全性高
  + torch.view只返回视图，不复制数据，结果保持连续性；torch.reshape尽量返回视图，在张量在内存中不连续时复制数据形成连续张量
  
  torch.transpose与torch.permute都是交换张量维度

  + torch.transpose只交换两个维度，复杂维度交换需要多次调用；torch.permute可一次性实现复杂维度变换
  + 二者都破坏连续性，都返回视图

  从张量的元数据的角度理解上述操作的区别

  + 每个torch中的张量都有一些元数据如shape, stride, dtype, device, data_ptr, is_contiguous等，内存占用很小
  + 视图的本质：复制元数据，共享数据体
  + 数据连续性的充要条件：对于所有维度 i，stride[i] = stride[i+1] × shape[i+1] 且最后一维 stride = 1，即逻辑索引=物理索引

### 原生tensor维度变换的缺点与einops的优势

  原生tensor维度变换操作的缺点

  + 索引硬编码，如举例的Transformer多头分割维度变换，交换维度需要记忆0213这种数字，缺乏语义
  + 链式操作复杂，尤其是需要不同方法连续使用的时候，代码长并且需要一步一步看在做什么
  + 代码只写做了什么，需要额外注释为什么
  
  einops的优势

  + 代码包含语义，不需额外的注释或者文档
  + 复杂不同类的维度变换操作一步到位

## 嵌入层的本质

每一个token对应一个词汇表中特定的Token ID

初始化嵌入矩阵后，词嵌入实际上是从嵌入矩阵中，根据Token ID取词嵌入向量再合并为矩阵

查表可根据索引直接访问，但是左乘one-hot矩阵包含矩阵乘法，即使是稀疏矩阵也比查表慢

## 前向传播与反向传播

### 前向传播

单个模块的前向传播过程，包含若干线性变换和非线性激活函数

以transformer的单个FFN层为例，包含一次扩张维度线性变换、一次非线性激活函数和一次压缩维度线性变换

前向传播不仅要计算这些过程的结果，还要为反向传播**保存中间结果**

### 反向传播

前向传播过程包含一次线性变换和一次ReLU激活函数
$$ \mathbf{z} = \mathbf{x} \mathbf{W}^T + \mathbf{b} $$
$$ \mathbf{y} = \text{ReLU}\left(\mathbf{z}\right) $$
反向传播会收到上一层传来的梯度
$$ \frac{\partial L}{\partial y_i}$$
现在求
$$ \frac{\partial L}{\partial W_{i,j}},\frac{\partial L}{\partial \mathbf{b}_j},\frac{\partial L}{\partial x_i},\frac{\partial L}{\partial z_j}$$
1.首先求$\frac{\partial L}{\partial z_j}$
$$ \frac{\partial L}{\partial z_j} = \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial z_j} $$
而ReLU函数的在$\mathbf{z}_j$小于等于0时，输出全为0，而大于0时保持原样输出

ReLU函数的导数在小于等于0时，导数为0；大于0时，导数为1

将下标拓展到整个$\mathbf{z}$实际上可以用前向传播保存的mask来计算
$$ \frac{\partial L}{\partial \mathbf{z}}= \frac{\partial L}{\partial \mathbf{y}} \cdot \text{mask}$$

2.再求 $\frac{\partial L}{\partial b_j}$
$$ \frac{\partial L}{\partial b_j}=\frac{\partial L}{\partial z_j}\cdot \frac{\partial z_j}{\partial b_j}=\frac{\partial L}{\partial z_j}$$
注意偏置$\mathbf{b}$的维度需要对齐$\mathbf{x}\mathbf{W}^T$，所以需要将$\mathbf{b}$
沿着batch方向广播到所有维度，对$\mathbf{b}$来说，它的维度从$(d_{out})$变更为$(N,d_{out})$