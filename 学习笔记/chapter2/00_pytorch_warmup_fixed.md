# pytorch warmup学习笔记

- [张量维度变换](#tensor-dimension)
  - [基本功：原生tensor维度变换操作复习与拓展](#basic-reshape)
  - [原生tensor维度变换的缺点与einops的优势](#einops-advantage)
- [嵌入层的本质](#embedding-essence)
- [前向传播与反向传播](#forward-backward)
  - [前向传播](#forward-propagation)
  - [反向传播](#back-propagation)

<a id="tensor-dimension"></a>
## 张量维度变换

<a id="basic-reshape"></a>
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

<a id="einops-advantage"></a>
### 原生tensor维度变换的缺点与einops的优势

原生tensor维度变换操作的缺点

+ 索引硬编码，如举例的Transformer多头分割维度变换，交换维度需要记忆0213这种数字，缺乏语义
+ 链式操作复杂，尤其是需要不同方法连续使用的时候，代码长并且需要一步一步看在做什么
+ 代码只写做了什么，需要额外注释为什么

einops的优势

+ 代码包含语义，不需额外的注释或者文档
+ 复杂不同类的维度变换操作一步到位

<a id="embedding-essence"></a>
## 嵌入层的本质

每一个token对应一个词汇表中特定的Token ID

初始化嵌入矩阵后，词嵌入实际上是从嵌入矩阵中，根据Token ID取词嵌入向量再合并为矩阵

查表可根据索引直接访问，但是左乘one-hot矩阵包含矩阵乘法，即使是稀疏矩阵也比查表慢

<a id="forward-backward"></a>
## 前向传播与反向传播

<a id="forward-propagation"></a>
### 前向传播

单个模块的前向传播过程，包含若干线性变换和非线性激活函数

以transformer的单个FFN层为例，包含一次扩张维度线性变换、一次非线性激活函数和一次压缩维度线性变换

前向传播不仅要计算这些过程的结果，还要为反向传播**保存中间结果**

中间结果利用ctx对象保存在GPU显存中(VRAM)

保存中间结果的原因是**用空间换时间**，一般来说存小结果如mask而不存结果 $\mathbf{z}$

<a id="back-propagation"></a>
### 反向传播

前向传播过程包含一次线性变换和一次ReLU激活函数

$$ \mathbf{z} = \mathbf{x} \mathbf{W}^T + \mathbf{b} $$

$$ \mathbf{y} = \text{ReLU}\left(\mathbf{z}\right) $$

反向传播会收到上一层传来的梯度

$$ \frac{\partial L}{\partial y_i} $$

现在求

$$ \frac{\partial L}{\partial W_{i,j}},\frac{\partial L}{\partial \mathbf{b}_j},\frac{\partial L}{\partial x_i},\frac{\partial L}{\partial z_j} $$

#### 1. 求 $\frac{\partial L}{\partial z_j}$

$$ \frac{\partial L}{\partial z_j} = \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial z_j} $$

而ReLU函数的在 $\mathbf{z}_j$ 小于等于0时，输出全为0，而大于0时保持原样输出

ReLU函数的导数在小于等于0时，导数为0；大于0时，导数为1

将下标拓展到整个 $\mathbf{z}$ 实际上可以用前向传播保存的mask来计算

$$ \frac{\partial L}{\partial \mathbf{z}}= \frac{\partial L}{\partial \mathbf{y}} \cdot \text{mask}$$

#### 2. 求 $\frac{\partial L}{\partial b_j}$

$$ \frac{\partial L}{\partial b_j}=\frac{\partial L}{\partial z_j}\cdot \frac{\partial z_j}{\partial b_j}=\frac{\partial L}{\partial z_j} $$

注意偏置 $\mathbf{b}$ 的维度需要对齐 $\mathbf{x}\mathbf{W}^T$，所以需要将 $\mathbf{b}$ 沿着batch方向广播到所有维度，对 $\mathbf{b}$ 来说，它的维度从 $(d_{out})$ 变更为 $(N,d_{out})$

我们最终计算导数也需要考虑广播，所以

$$ \frac{\partial L}{\partial \mathbf{b}}=\sum_{i=1}^N \frac{\partial L}{\partial \mathbf{z}} $$

写成代码即为

```python
grad_bias = grad_z.sum(dim=0)
```

因为广播实际是在第0个维度进行的

#### 3. 求 $\frac{\partial L}{\partial W_{i,j}}$

$$ \frac{\partial L}{\partial W_{i,j}} = \frac{\partial L}{\partial z_j}\cdot\frac{\partial z_j}{\partial W_{i,j}} $$

$$ z_j = \sum_kx_kW_{k,j} + b_j $$

所以 $\frac{\partial z_j}{\partial W_{i,j}}=x_i$，带入可得

$$ \frac{\partial L}{\partial W_{i,j}} = \frac{\partial L}{\partial z_j} \cdot x_i $$

当样本数为 $N$ 时我们有

$$ \frac{\partial L}{\partial W_{i,j}}=\sum_n\frac{\partial L}{\partial Z_{n,j}}\cdot \mathbf{X}_{n,i} $$

我们固定i对j遍历，注意到上面的式子实际是列向量的内积的展开形式，写成矩阵形式即为

$$ \frac{\partial L}{\partial W_{i,:}}=\left( \left( \frac{\partial L}{\partial Z}\right)_{:,i}\right)^T \cdot \mathbf{X}$$

再对i遍历我们有

$$ \frac{\partial L}{\partial W} = \left( \frac{\partial L}{\partial Z}\right)^T \cdot \mathbf{X}$$

写成代码即为

```python
grad_weight = grad_z.T @ x
```

这里为了体现 $\mathbf{z}$ 的维度数临时写作 $\mathbf{Z}$

#### 4. 求 $\frac{\partial L}{\partial x_i}$

这一部分跟上面对权重求导类似，这里省略推导了

$$ \frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{z}}\cdot \mathbf{W} $$

写成代码即为

```python
grad_x = grad_z @ weight
```
