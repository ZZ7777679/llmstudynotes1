# pytorch warmup学习笔记

- [pytorch warmup学习笔记](#pytorch-warmup学习笔记)
  - [张量维度变换](#张量维度变换)
    - [基本功：原生tensor维度变换操作复习与拓展](#基本功原生tensor维度变换操作复习与拓展)
    - [原生tensor维度变换的缺点与einops的解决方案](#原生tensor维度变换的缺点与einops的解决方案)
  - [嵌入层的本质](#嵌入层的本质)
  - [前向传播与反向传播](#前向传播与反向传播)
  
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

### 原生tensor维度变换的缺点与einops的解决方案

  原生tensor维度变换操作的缺点

  + 索引硬编码，如举例的Transformer多头分割维度变换，交换维度需要记忆0213这种数字，缺乏语义
  + 链式操作复杂，尤其是需要不同方法连续使用的时候，代码长并且需要一步一步看在做什么
  + 代码只写做了什么，需要额外注释为什么
  
  einops的优势

  + 代码包含语义，不需额外的注释或者文档
  + 复杂不同类的维度变换操作一步到位

## 嵌入层的本质

## 前向传播与反向传播