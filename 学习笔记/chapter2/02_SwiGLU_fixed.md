# SwiGLU

- [SwiGLU](#swiglu)
  - [Swish激活函数](#swish)
  - [GLU(门控线性单元)](#glu)
  - [SwiGLU](#swiglu-1)
  - [SwiGLU深入](#swiglu)
    - [路由规则](#section)
    - [数据依赖](#section-1)
    - [决策与信息解析解耦](#section-2)
    - [综合不同特征维度的门控信号](#section-3)
    - [更强的数学表达能力](#section-4)
    - [更健康的梯度流](#section-5)
  - [SwiGLU的参数量对齐](#swiglu-1)
    - [原始FFN层参数量](#ffn)
    - [含SwiGLU的FFN层参数量](#swiglu-2)
  - [工业级实现与性能陷阱](#section-6)
    - [张量并行与内存对齐](#section-7)
    - [访存瓶颈与算子融合](#section-8)

<a id="swish"></a>
## Swish激活函数

Swish是一种自门控激活函数

$$ \text{Swish}_{\beta}\left(x\right) = x \cdot \sigma\left( \beta x\right) = x \cdot \frac{1}{1+e^{-\left(\beta x\right)}}$$

当 $\beta = 1$ 时，Swish激活函数退化为SiLU激活函数：

$$ \text{SiLU} \left( x \right) = x \cdot \sigma \left( x \right) $$

对于一个自门控激活函数 $f \left( x \right) = x \cdot \sigma \left( x \right)$，它由两部分组成
+ 输入 $x$
+ 门控信号 $\sigma \left( x \right)$

什么叫“自门控”：门控信号 $\sigma \left( x \right)$ 的参数来源就是输入 $x$ 本身
自门控激活函数有什么特点：
+  当输入x→+∞时，门控信号趋近于1，输入无损通过
+  当输入x→-∞时，门控信号趋近于0但不为严格的0，输入被抑制并且梯度仍能流动
+  输入在x→ 0时，门控处于平滑过渡区，保留微弱的负梯度
+  不需要额外的权重矩阵来生成门控信号

<a id="glu"></a>
## GLU(门控线性单元)

GLU，即门控线性单元的公式定义

$$ \text{GLU}\left( x,W,V,b,c \right) = \sigma\left(xW+b\right) \odot \left( x V + c \right) $$

其中 $\odot$ 是矩阵逐元素乘法，又被称为Hadamard积
这个公式由两个部分组成
+ 门控分支 $\sigma \left( x W + b \right)$：经过一个sigmoid激活函数，输出介于0-1的门控信号
+ 数值分支 $\left( x V + c \right)$：进行一次线性变换

最终输出 = 门控分支 × 数值分支

## SwiGLU

SwiGLU是将Swish作为GLU门控分支的激活函数的变体，其公式为

$$ \text{SwiGLU} \left( x, W, V, b, c, \beta \right) = \text{Swish}_{\beta}\left(x W + b \right) \odot \left( x V + c \right) $$

当 $\beta = 1$ 时，公式简化为

$$ \text{SwiGLU} \left( x, W, V, b, c \right) = \text{SiLU} \left( x W + b \right) \odot \left( x V + c \right) $$

实际实现中为了简化运算会去掉偏置项，所以公式进一步简化为

$$ \text{SwiGLU} \left(x, W,V \right) = \text{SiLU} \left(x W \right) \odot \left( x V \right) $$

在Transformer中，SwiGLU是用在FFN层中的，结合FFN层的维度膨胀与收缩操作

$$ \text{FFN}_{\text{SwiGLU}} \left( x \right) = W_{Down} \cdot \left( \text{SiLU} \left(x W_{gate} \right)\odot \left(x W_{Up}\right)\right) $$

<a id="swiglu"></a>
## SwiGLU深入

<a id="section"></a>
### 路由规则
+ SwiGLU的路由规则是连续可调的软开关，决策边界会随着输入数值变化而动态地、平滑地变化
+ ReLU的路由规则是预先写好边界的硬阈值开关，决策边界**不会**随着输入数值变化而变化

<a id="section-1"></a>
### 数据依赖
+ SwiGLU的路由依据是输入经过可学习的门控权重矩阵变换后的高维空间投影，具有全局视野
+ ReLU的路由依据是孤立的输入本身

<a id="section-2"></a>
### 决策与信息解析解耦
SwiGLU的门控信号来自 $x W_{gate}$，而数值内容来自 $x W_{Up}$
而 $x W_{gate}$ 和 $x W_{Up}$ 是相互独立的权重矩阵
这代表着路由决策与信息解析两件事情解耦了，可以独立优化

<a id="section-3"></a>
### 综合不同特征维度的门控信号
$x W_{gate}$ 实际上把向量 $x$ 的所有维度进行了线性加权求和，在维度的意义上，这个操作引入了关于特征维度的上下文信息，组合了所有维度的数值，组合出了一个门控信号

<a id="section-4"></a>
### 更强的数学表达能力
另外，SwiGLU是用了矩阵逐项乘，这引入了二阶交叉项，相比于ReLU的一阶线性分割，使用SwiGLU的FFN层直接就具备了拟合非线性特征交互的能力

<a id="section-5"></a>
### 更健康的梯度流
ReLU负半轴阶段，这让梯度直接为0，会引起神经元死亡，梯度无法传递
SwiGLU的SiLU门控在负半轴保留了微弱但非零的梯度，确保梯度能够传递，训练更稳定

<a id="swiglu-1"></a>
## SwiGLU的参数量对齐

<a id="ffn"></a>
### 原始FFN层参数量

- 维度膨胀操作：维度从 $d_{model}$ 拓展到 $d_{ff} = 4 d_{model}$
- 维度压缩操作：维度从 $d_{ff}$ 压缩到 $d_{model}$
- 参数量计算：

$$ 2 \times \left(d_{model} \times d_{ff} \right) = 2 \times \left( d_{model} \times 4 \times d_{model} \right) = 8 d_{model}^2 $$

<a id="swiglu-2"></a>
### 含SwiGLU的FFN层参数量

除了膨胀矩阵和压缩矩阵，SwiGLU中新增了一个门控权重矩阵，所以参数量为

$$ 3 \times \left( d_{model} \times d_{ff} \right)$$

若想要参数量根原来相等，则

$$ 3 \times \left( d_{model} \times d_{ff} \right) = 8 d_{model}^2$$

所以

$$ d_{ff} = \frac{8}{3} d_{model} $$

所以理论上这个维度应为 $d_{model} \times \frac{8}{3} = 4096 \times \frac{8}{3} \approx 10922.67$，而工程上向上取整到256的倍数，所以最后得到的中间层维度数为11008

<a id="section-6"></a>
## 工业级实现与性能陷阱


<a id="section-7"></a>
### 张量并行与内存对齐

先来回答上一部分最后留下的问题，为什么是256的倍数？
+ CUDA内存对齐，cudaMalloc()分配内存默认对齐到256字节
+ 一些特定的GPU操作以256字节为增量进行
+ 单卡的内存事务优化中，GPU以32字节固定大小的单位进行内存事务。一个线程束(32个线程)读32×4=128B的内存，如果访问连续对齐的256字节内存段，则硬件能用最少的访存事务高效完成
+ 在多卡的情况下，大模型训练时会将这个中间维度拆分到不同的卡中并行运算，所以这个维度要支持被卡数量整除，而256的倍数正好满足这个条件
+ Tensor Core作为加速矩阵乘法的专用单元，对矩阵维度有要求。如FP8精读下要求宽度是16的倍数，256正好满足这个条件

<a id="section-8"></a>
### 访存瓶颈与算子融合

如果按照计算步骤依次计算，则FFN层包含三次运算：
门控信号、矩阵膨胀、矩阵压缩

当输入张量非常大时，在门控信号计算和矩阵膨胀计算中，会连续读取同样的输入张量，甚至于读数据的时间超过了矩阵乘法本身的计算时间

为了解决这个问题，工业上会将这两个权重矩阵拼成一个大矩阵，大矩阵去于x做矩阵乘法，计算完之后再分开，这样做可以
+ 融合两个操作，让输入被读取的次数从2次降为1次，减少50%的显存读取开销
+ 几乎0开销，只需要多一次矩阵拼接和一次矩阵拆分
+ 最终大概加速比为1.3x~1.5x