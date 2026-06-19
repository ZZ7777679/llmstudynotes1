# RoPE

- [RoPE](#rope)
  - [从绝对位置编码到相对位置编码](#position-encoding)
    - [为什么要用位置编码](#why-position-encoding)
    - [绝对位置编码与其局限性](#absolute-position-limitations)
    - [传统相对位置编码](#traditional-relative-position)
  - [RoPE](#rope-overview)
    - [RoPE的核心设计思想](#rope-core)
    - [为什么是旋转且匀速](#why-rotation)
    - [RoPE的复平面推导](#rope-complex-plane)
      - [向量与复数的映射](#vector-complex-mapping)
      - [旋转矩阵的复数等价形式](#complex-rotation)
      - [复平面中的内积表示](#complex-inner-product)
      - [复平面中RoPE的相对性](#rope-relativity)
      - [复平面中RoPE的几何意义](#rope-geometry)
  - [RoPE的代码实现](#rope-code)
  - [旋转位置编码旋转频率的意义](#rotation-frequency)
    - [为什么不用单一旋转速度](#why-multiple-frequencies)
    - [旋转的高维拓展](#high-dimensional-extension)

<a id="position-encoding"></a>
## 从绝对位置编码到相对位置编码

<a id="why-position-encoding"></a>
### 为什么要用位置编码

Transformer的自注意力本质上是计算两两向量的点积

点积运算具有交换律，这意味着模型将输入当作一个无序的集合

但是自然语言显然不是无序的，同样的语素交换顺序之后意义可能天差地别，所以需要给词嵌入向量增加位置信息，这一功能就是由位置编码实现的

<a id="absolute-position-limitations"></a>
### 绝对位置编码与其局限性

绝对位置编码给每一个位置 $m$ 分配一个唯一的向量 $\mathbf{p}_m$，如最经典的正弦编码

$
\mathbf{PE}_{\left(m,2i\right)}=\sin \left(\frac{m}{10000^{\frac{2i}{d}}}\right),\quad  \mathbf{PE}_{\left(m,2i+1\right)}=\cos \left(\frac{m}{10000^{\frac{2i}{d}}}\right)
$

但是绝对位置编码有两个问题

+ 外推灾难

当token位置 $m\le\mathbf{L}_{train}$ 即最大训练长度时，模型见过这些 $m$ 的取值，模型表现正常

当token位置 $m\gt\mathbf{L}_{train}$ 即最大训练长度时，模型面对的是分布外的输入值，根据正弦编码计算公式，正弦余弦的弧度会落入训练集未出现过的区域，这样生成的结果会偏离训练时的均值和方差

进一步地，这些异常值在层归一化过程中计算的均值和方差剧烈抖动，softmax函数会因为这些极端输入让注意力趋于one-hot或者平坦，最终导致语义崩坏

+ 平移不变性的缺失

在自然语言中，相邻这种关系比绝对坐标更重要

但是绝对位置编码对不同绝对位置的token对的位置编码是截然不同的

对于不同位置的相邻关系，模型需要根据绝对位置编码从头学习它们的这种关系

它学不会类似"相距为1的两个词应当拥有相近的注意力模式"这种通用的语言规律

在训练中，这种缺陷会让模型被迫去重复学习绝对位置不同的、相对位置相关的语言规律，导致模型参数量巨大但是泛化能力却并不强

在推理中，如果遇到了训练中少见的绝对位置上的局部关系，模型甚至可能不能解析这种关系，难以维持长文本的逻辑连贯性

<a id="traditional-relative-position"></a>
### 传统相对位置编码

+ 偏置类相对位置编码的KV缓存失效问题

在RoPE之前提出了一种相对位置偏置的位置编码，思路是在计算注意力分数时额外配置一个依赖距离的偏置项

$
 \text{Score} = \mathbf{q}_m \cdot \mathbf{k}_n + b_{m-n} 
$

但是相对位置会随着当前总长度的变化而变化，这样每一次生成新词时都需要遍历整个KV cache，动态地计算新的偏置矩阵b

这样反而导致引入大量的浮点运算量和显存带宽瓶颈，同时动态偏置的引入会导致无法使用FlashAttention，所以这种方法实际应用较少

<a id="rope-overview"></a>
## RoPE

<a id="rope-core"></a>
### RoPE的核心设计思想

+ RoPE设计的目标需要能解决前述三大痛点
  + 必须有平移不变性
  + 必须有强外推能力
  + 不能碰KV cache

最终转化为这样一个问题：

是否存在一个函数 $g$，使得 $\langle \mathbf{Q}_m , \mathbf{K}_n \rangle= g\left(m-n\right)$

<a id="why-rotation"></a>
### 为什么是旋转且匀速

首先在2D平面上找直觉

我们的目标是，找到一个关于位置 $m$ 的线性变换 $\mathbf{T}_m$ 使得变换后的向量内积只依赖于相对距离 $m-n$

设原始查询向量为 $\mathbf{q}$，键向量为 $\mathbf{k}$，经过位置编码后得到：

$
\mathbf{q}_m=\mathbf{T}_m\mathbf{q},\quad\mathbf{k}_n=\mathbf{T}_n\mathbf{k}
$

而我们的要求是这个变换需要满足：

$
\langle\mathbf{q}_m,\mathbf{k}_n\rangle=\langle\mathbf{T}_m\mathbf{q},\mathbf{T}_n\mathbf{k}\rangle=g\left(m-n\right)
$

将内积展开为矩阵乘法形式可得：

$
\langle\mathbf{T}_m\mathbf{q},\mathbf{T}_n\mathbf{k}\rangle=\mathbf{q}^T\left(\mathbf{T}_m^T\mathbf{T}_n\right)\mathbf{k}
$

由于上式子对任意 $\mathbf{q},\mathbf{k}$ 都成立，而等式右边 $g\left(m-n\right)$ 依赖于 $m-n$，所以中间矩阵 $\mathbf{T}_m^T\mathbf{T}_n$ 也依赖于 $m-n$

所以必然存在一个矩阵序列 $\mathbf{R}_{\Delta}$ 使得

$
\mathbf{T}_m^T\mathbf{T}_n=\mathbf{R}_{m-n}
$


这个矩阵序列的意义是描述两个不同位置编码函数之间相互作用的结果

当 $m=n$ 时，我们可以得到

$
\mathbf{T}_m^T\mathbf{T}_m=\mathbf{R}_0
$

而左侧向量与自身的内积数学计算结果应为模长的平方，而它的意义是变换 $\mathbf{T}_m$ 对向量模长的影响

而我们的位置编码的职责是**只注入位置信息，不篡改语义强度**，所以我们要求 $\mathbf{R}_0=\mathbf{I}$

这得到一个结论：任意位置 $m$ 的位置编码变换矩阵 $\mathbf{T}_m$ 满足 $\mathbf{T}_m^T\mathbf{T}_m=\mathbf{I}$

这样的的矩阵被称为正交矩阵，它的几何意义是保距变换，即只改变方向，不改变大小

在二维平面中这样的变换只有两种可能：

+ 旋转，位置变换矩阵行列式值为1
+ 镜像反射，位置变换行列式值为-1

当位置为0时，显然此时没有附加位置编码，$\mathbf{T}_0=\mathbf{I}$，其行列式值为1

当位置参数 $m$ 变化时，矩阵元素随着 $m$ 连续变化，所以位置变换矩阵的行列式也是连续的

但是我们又知道位置变换矩阵的行列只能为+1或-1，且起始点的行列式为1，所以它不能是-1

所以，这个位置变换一定是旋转

进一步的，**这个旋转一定是匀速的**

我们已经推导出这个位置变换是旋转了

那么我们假设旋转角度为 $\phi\left(m\right)$，那么 $\mathbf{T}_m$ 旋转了 $\phi\left(m\right)$ 弧度

由 $\mathbf{T}_m^T\mathbf{T}_n=\mathbf{R}_{m-n}$ 可知，而左侧 $\mathbf{T}_m^T$ 的几何意义为 $\mathbf{T}_m$ 的反向旋转操作，这个等式说明左侧只依赖于 $m-n$，即

$
\phi\left(n\right)-\phi\left(m\right)=g\left(m-n\right)
$

取 $m=0$ 得 $\phi\left(n\right)-\phi\left(0\right)=g\left(-n\right)$

而 $\phi\left(0\right)=0$，所以 $g\left(n\right)=\phi\left(-n\right)$

所以 $\phi\left(n\right)-\phi\left(m\right)=g\left(m-n\right)=\phi\left(n-m\right)$

解得 $\phi\left(m\right)=m\theta$

所以我们可以得到最终的位置 $m$ 的变换矩阵为

$$
\mathbf{T}_m=
\begin{pmatrix}
    \cos\left(m\theta\right) & -\sin\left(m\theta\right)\\
    \sin\left(m\theta\right) & \cos\left(m\theta\right)
\end{pmatrix}
$$

<a id="rope-complex-plane"></a>
### RoPE的复平面推导

<a id="vector-complex-mapping"></a>
#### 向量与复数的映射

在二维平面中，我们将向量 $(x, y)$ 映射为一个复数：

$
 \mathbf{v} = (x, y) \quad \longleftrightarrow \quad z = x + iy 
$

其中 $i^2 = -1$。

几何意义：复平面的实轴对应 $x$ 轴，虚轴对应 $y$ 轴。一个复数本质上就是平面上的一个点（或一个向量）。

<a id="complex-rotation"></a>
#### 旋转矩阵的复数等价形式

我们已知位置 $m$ 的变换矩阵为：

$
 \mathbf{T}_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} 
$

将一个向量 $(x, y)$ 左乘 $\mathbf{T}_m$，得到旋转后的坐标 $(x', y')$：

$
 x' = x\cos(m\theta) - y\sin(m\theta) 
$

$
 y' = x\sin(m\theta) + y\cos(m\theta) 
$

构造对应的复数 $z' = x' + iy'$：

$
 z' = \big[x\cos(m\theta) - y\sin(m\theta)\big] + i\big[x\sin(m\theta) + y\cos(m\theta)\big] 
$

提取公因式并整理：

$
 z' = x\big[\cos(m\theta) + i\sin(m\theta)\big] + y\big[-\sin(m\theta) + i\cos(m\theta)\big] 
$

由于 $i \cdot i = -1$，我们可以将第二项的系数改写为 $i y$ 乘以 $(\cos + i\sin)$，即：

$
 z' = (x + iy)\big[\cos(m\theta) + i\sin(m\theta)\big] 
$

根据欧拉公式 $e^{i\phi} = \cos\phi + i\sin\phi$，得到：

$
 z' = z \cdot e^{im\theta} 
$

几何意义：复数乘法天然就是“旋转 + 缩放”

而 $e^{im\theta}$ 的模长恒为1，它只负责让向量在复平面上旋转 $m\theta$ 弧度，而绝对不改变向量的模长(语义强度)，这也是与之前的保距变换要求相对应。

<a id="complex-inner-product"></a>
#### 复平面中的内积表示

在实数空间中，两个向量 $\mathbf{q}$ 和 $\mathbf{k}$ 的内积（点积）定义为：

$
 \langle \mathbf{q}, \mathbf{k} \rangle = q_x k_x + q_y k_y 
$

在复数域中，如果我们设这两个向量对应的复数为 $q$ 和 $k$，则该内积等价于：

$
 \langle \mathbf{q}, \mathbf{k} \rangle = \text{Re}\big(q \cdot \overline{k}\big) 
$

其中 $\overline{k}$ 表示 $k$ 的共轭复数（即虚部取反），$\text{Re}$ 表示取结果的实部。

验证：设 $q = a+bi$，$k = c+di$，则 $q \cdot \overline{k} = (a+bi)(c-di) = (ac+bd) + i(bc - ad)$，其实部正好是 $ac+bd$，完美对应实数向量的点积公式。

<a id="rope-relativity"></a>
#### 复平面中RoPE的相对性

设原始查询和键对应的复数分别为 $q$ 和 $k$。

经过 RoPE 编码后：

- 位置 $m$ 的查询复数为：

$
 q_m = q \cdot e^{im\theta} 
$

- 位置 $n$ 的键复数为：

$
 k_n = k \cdot e^{in\theta} 
$

我们关心的是注意力机制中的内积 $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$，它在复数域中写作：

$
 \langle \mathbf{q}_m, \mathbf{k}_n \rangle = \text{Re}\big( q_m \cdot \overline{k_n} \big) 
$

将 $q_m$ 和 $k_n$ 代入：

$
 \langle \mathbf{q}_m, \mathbf{k}_n \rangle = \text{Re}\Big( \big(q \cdot e^{im\theta}\big) \cdot \overline{\big(k \cdot e^{in\theta}\big)} \Big) 
$

由于共轭运算将指数取反（即 $\overline{e^{in\theta}} = e^{-in\theta}$），上式变为：

$
 \langle \mathbf{q}_m, \mathbf{k}_n \rangle = \text{Re}\Big( q \cdot e^{im\theta} \cdot \overline{k} \cdot e^{-in\theta} \Big) 
$

合并指数部分：

$
 \langle \mathbf{q}_m, \mathbf{k}_n \rangle = \text{Re}\Big( q \cdot \overline{k} \cdot e^{i(m-n)\theta} \Big) \tag{1} 
$

令 $S = q \cdot \overline{k}$（这是一个固定的复数，与位置无关），则最终结果为：

$
 \langle \mathbf{q}_m, \mathbf{k}_n \rangle = \text{Re}\Big( S \cdot e^{i(m-n)\theta} \Big) \tag{2} 
$

这样可以得到结论：公式 (2) 中，位置信息仅以 $(m-n)$ 的差值形式出现，而不再含有单独的 $m$ 或 $n$。也就是说，**绝对位置被消除，内积只依赖于相对距离**——这也是旋转位置编码设计的核心

<a id="rope-geometry"></a>
#### 复平面中RoPE的几何意义

公式 (2) 的几何含义非常直观，可以分两层理解：

- 第一层：$S = q \cdot \overline{k}$ 包含了原始向量之间的夹角信息（即没有位置编码时的内积）。
- 第二层：乘以 $e^{i(m-n)\theta}$ 相当于在原始夹角的基础上，额外增加了一个大小为 $(m-n)\theta$ 的相位偏移。

换句话说：

- $q$ 被旋转了 $m\theta$（位置 $m$ 的贡献）；
- $k$ 被旋转了 $n\theta$（位置 $n$ 的贡献）；
- 两个旋转后的向量之间的夹角 = 原始夹角 $+ (m-n)\theta$。

而内积 $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$ 计算的是这个夹角的余弦值，因此它天然地携带了相对距离 $(m-n)$ 的信息。

几何意义：想象两个指针在表盘上以不同的速度转动。查询指针转了 $m\theta$ 度，键指针转了 $n\theta$ 度。我们并不关心它们各自转到了哪个绝对刻度，只关心它们之间的**夹角差**——而这个夹角差恰好正比于 $(m-n)$，即它们位置的距离。

<a id="rope-code"></a>
## RoPE的代码实现

+ 这个好像实现比我想象的难多了...公式理解了记好了，会写又是另一回事了

第一部分是计算旋转角，这里比较麻烦的是怎么表示频率常数，这里需要自己仔细推一推怎么跟代码对上

然后再依次生成位置序列，计算外积，最后转换为复数形式，为后面复数运算做准备

第二部分是广播形状对齐

想要与输入张量进行矩阵运算，需要把旋转因子的维度进行一个前向的填充，让batch维度和num_heads维度能自动广播，seq_len和head_dim对齐

第三部分是最后的旋转应用

这里主要是负数与实数的互相转换，主要理解了复数的计算这里就还好

+ 为什么用复数
  + GPU对复数运算有指令优化，并且减少了中间张量的显存占用
  + 数学公式到代码的映射更加直观

<a id="rotation-frequency"></a>
## 旋转位置编码旋转频率的意义

<a id="why-multiple-frequencies"></a>
### 为什么不用单一旋转速度

+ 信息冗余

高维空间提供了巨大的表达通道，只用一个旋转速度会浪费这种高维表征能力。

+ 无法同时满足局部敏感和远程稳定

如果旋转速度很高，那么对相邻距离会非常敏感，但是当距离变大时，相位差很容易就超过180°。此时余弦函数失去单调性，内积不再随距离增大而减小，反而开始振荡回升，甚至在接近360°时重新达到峰值，出现距离远但注意力分数上升的问题。若继续增大至超过360°，还会出现完全的归位混叠，但在此之前模型性能早已因非单调性而崩溃。

如果旋转速度很低，那么对较远距离可以合理描述，但是对于相邻距离来说，过低的转速导致相位差极小，在浮点数精度下难以区分，模型无法有效捕捉“相邻”这种最重要的局部位置信息。

<a id="high-dimensional-extension"></a>
### 旋转的高维拓展

当特征维度索引较小时，旋转速度快，擅长捕捉局部位置信息。

当特征维度索引较大时，旋转速度慢，擅长捕捉长程位置信息。

二者结合，高频分量在近距离提供尖锐的区分度，在远距离因振荡而相互抵消；低频分量则全程提供平滑的远程锚定。多频率加权叠加后，使得注意力分数随相对距离呈现出理想的“单调递减且平滑”的远程衰减特性，这样既能在短距离内极其敏感，又能在长距离上持续不混乱。
