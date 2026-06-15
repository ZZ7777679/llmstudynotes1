# RMSNorm的实现

- [RMSNorm的实现](#rmsnorm的实现)
  - [核心思想](#核心思想)
  - [具体公式与实现](#具体公式与实现)
  - [代码实现细节](#代码实现细节)


## 核心思想

与过去的LayerNorm相比直接舍去了均值计算，实际上被证明效果与原来有均值计算相当甚至更好

## 具体公式与实现

+ 计算均方根(RMS)
  
$$ \mathbf{RMS}\left(x\right) = \sqrt{\frac{1}{d}\sum_{i=1}^dx_i^2+\epsilon}$$

+ 归一化并缩放(Scale)
  
  注意**只缩放，不平移**

$$ \mathbf{RMSN}\left(x\right) = \frac{x}{\mathbf{RMS}\left(x\right)}\cdot\gamma$$

+ 与原始LayerNorm对比
  
原始LayerNorm公式如下：

$$ \mathbf{LN}\left(x\right) = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\cdot\gamma+\beta$$

其中

$$ \mu=\frac{1}{d}\sum_{i=1}^dx_i \quad , \quad \sigma=\sqrt{\frac{1}{d}\sum_{i=1}^d(x_i-\mu)^2+\epsilon} $$

假设特征维度为d，计算运算次数，则
对于LayerNorm，总计算开销约为 $ 8d+O(1) $
对于RMSNorm，总计算开销约为 $ 4d+O(1) $

可以得到结论：RMSNorm计算开销是LayerNorm的一半左右

+ 混合精度陷阱
  
  现在常用混合精度，但是进行平方计算容易数值超过该精度的数值范围，所以这里需要转换精度进行平方计算，完成RMSNorm后再将精度压回去

## 代码实现细节

+ 先转换到高精度，这个if判定其实不是必须的，.to()已经做过内部优化了
  
+ 求均值要在最后一个维度进行
  + 求均值后最后一个维度为1，后续广播从右向左判定发现有1，将最后一个维度广播，从而保持维度一致
  + 在最后一个维度求均值，也是对每一个token的特征向量独立地归一化，否则会跨batch共享一个RMS值

+ 有torch.rsqrt更快则不用进行1/sqrt计算

+ RMSNorm完全计算完成后再转回原本的精度，每次计算只是一个临时内存开销，而长期存储的转换为低精度降低开销