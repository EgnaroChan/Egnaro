# KL Divergence（KL 散度）

## 1. KL 散度的定义

### 1.1 离散分布与连续分布

设 $P,Q$ 是同一离散样本空间 $\mathcal X$ 上的概率分布。**KL 散度（Kullback–Leibler divergence）**定义为

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_{x\in\mathcal X}
P(x)\log\frac{P(x)}{Q(x)}.
$$

等价地，可以写成对 $P$ 取期望的形式：

$$
D_{\mathrm{KL}}(P\|Q)
=\mathbb E_{x\sim P}\left[\log\frac{P(x)}{Q(x)}\right].
$$

如果两个分布在连续样本空间上具有相对于同一基准测度的密度 $p(x)$ 和 $q(x)$，则

$$
D_{\mathrm{KL}}(P\|Q)
=\int p(x)\log\frac{p(x)}{q(x)}\,dx.
$$

### 1.2 当某个事件的概率为零时

在离散情况下，零概率有两种不同情形：

- 如果 $P(x)=0$，这个事件不会从真实分布中出现，对求和没有贡献；约定对应项为 $0$。
- 如果 $P(x)>0$，但 $Q(x)=0$，模型把一个实际可能发生的事件判断为绝不可能，损失 $-\log Q(x)=+\infty$，因此 $D_{\mathrm{KL}}(P\|Q)=+\infty$。

上述零概率条件可以用一个标准记号表示：**绝对连续（absolute continuity）**

$$
P\ll Q.
$$

它的严格含义是：**只要 $Q$ 认为某个事件集合 $A$ 的概率为零，$P$ 也必须认为它的概率为零**，即

$$
Q(A)=0\quad\Longrightarrow\quad P(A)=0.
$$

对于有限离散样本空间，这恰好等价于

$$
P(x)>0\quad\Longrightarrow\quad Q(x)>0.
$$

因此，KL 有限要求 $P\ll Q$：$Q$ 的零概率事件不能在 $P$ 下有正概率。**在一般连续空间中，$P\ll Q$ 只是 KL 有限的必要条件，不是充分条件**：对数密度比的期望仍可能发散。

## 2. 基本性质

### 2.1 非负性（Gibbs 不等式）

对任意两个概率分布 $P,Q$，

$$
D_{\mathrm{KL}}(P\|Q)\geq 0,
$$

且等号成立当且仅当 $P=Q$（连续情形为几乎处处相等）。这个结论说明 KL 的最小值是 $0$，但**不能理解为求和中的每一项都非负**：当 $P(x)<Q(x)$ 时，$P(x)\log(P(x)/Q(x))$ 就是负数。

**证明（离散情形）。** 若存在 $P(x)>0$、$Q(x)=0$，则 KL 为 $+\infty$。否则，利用 $-\log u$ 的凸性以及 Jensen 不等式，

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\mathbb E_{x\sim P}
\left[-\log\frac{Q(x)}{P(x)}\right]\\
&\geq-\log\mathbb E_{x\sim P}
\left[\frac{Q(x)}{P(x)}\right]\\
&=-\log\sum_{x:P(x)>0}Q(x)\\
&\geq-\log 1=0.
\end{aligned}
$$

最后一个不等式使用了 $\sum_{x:P(x)>0}Q(x)\leq1$。等号成立时，$Q$ 在 $P$ 的支撑集外没有概率质量，且由 Jensen 的严格凸性，$Q(x)/P(x)$ 在支撑集上为常数；归一化条件进一步要求这个常数等于 $1$。

### 2.2 不对称性

KL 一般不满足对称性。取

$$
P=(0.5,0.5),\qquad Q=(0.75,0.25).
$$

直接计算得到

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)&\approx0.1438,\\
D_{\mathrm{KL}}(Q\|P)&\approx0.1308.
\end{aligned}
$$

因此，$D_{\mathrm{KL}}(P\|Q)\neq D_{\mathrm{KL}}(Q\|P)$。其区别来自**取期望的分布不同**：前者按 $P(x)$ 对各个位置加权，后者按 $Q(x)$ 加权。KL 也一般不满足三角不等式，因而不是数学意义上的距离（metric）。

## 3. KL、交叉熵与最大似然估计

### 3.1 交叉熵与 KL 的关系

对于离散分布 $P$，熵（entropy）定义为

$$
H(P)=-\sum_xP(x)\log P(x).
$$

给定另一个分布 $Q$，交叉熵（cross-entropy）定义为

$$
H(P,Q)=-\sum_xP(x)\log Q(x).
$$

两者的期望分布都是 $P$，区别只在于对数项使用 $P(x)$ 还是 $Q(x)$。将 KL 的定义展开，

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\sum_xP(x)\log P(x)
-\sum_xP(x)\log Q(x)\\
&=H(P,Q)-H(P).
\end{aligned}
$$

这是一个精确恒等式。它也说明了 KL 的一个常用解释：**若样本服从 $P$，则使用 $Q$ 代替 $P$ 所增加的平均负对数概率，恰好等于 $D_{\mathrm{KL}}(P\|Q)$**。这里无需先引入编码理论，直接比较两个期望即可。

### 3.2 为什么最大似然对应 Forward KL？

设 $p_{\mathrm{data}}(x)$ 为真实数据分布，$p_\theta(x)$ 为参数为 $\theta$ 的模型分布。我们希望通过调整 $\theta$，使模型分布接近真实分布。根据 KL 的定义，

$
\begin{aligned}
D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)
&=\mathbb E_{x\sim p_{\mathrm{data}}}
\left[\log\frac{p_{\mathrm{data}}(x)}{p_\theta(x)}\right]\\
&=\mathbb E_{x\sim p_{\mathrm{data}}}\log p_{\mathrm{data}}(x)
-\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x)\\
&=-H(p_{\mathrm{data}})
-\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x).
\end{aligned}
$

这里 $p_{\mathrm{data}}$ 是固定的真实分布，不随参数 $\theta$ 改变，因此 $H(p_{\mathrm{data}})$ 对 $\theta$ 来说是常数。最小化 KL 时，可以忽略这个常数项：

$
\begin{aligned}
\arg\min_\theta D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)
&=\arg\min_\theta\left[
-\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x)
\right]\\
&=\arg\max_\theta
\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x).
\end{aligned}
$

也就是说，**最小化真实分布到模型分布的 Forward KL，等价于最大化模型在真实数据上的平均对数概率**。这里的期望始终是对 $p_{\mathrm{data}}$ 计算的，而不是对模型分布 $p_\theta$ 计算。

但真实分布 $p_{\mathrm{data}}$ 未知，右侧的期望无法直接计算。实际训练时，我们只有从它独立同分布采样得到的数据集 $\mathcal D=\{x^{(1)},\ldots,x^{(N)}\}$。

**首先看最大似然估计（MLE）本身的定义。** 在模型 $p_\theta$ 下，整批独立样本的联合似然是

$$
\mathcal{L}(\theta;\mathcal D)
=\prod_{i=1}^{N}p_\theta(x^{(i)}).
$$

MLE 选择使这批数据似然最大的参数。由于对数函数严格递增，最大化似然等价于最大化对数似然；取负号并除以样本数，又等价于最小化**平均负对数似然（NLL）**：

$$
\begin{aligned}
\hat\theta_{\mathrm{MLE}}
&=\arg\max_\theta
\sum_{i=1}^{N}\log p_\theta(x^{(i)})\\
&=\arg\min_\theta
\underbrace{\left(
-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x^{(i)})
\right)}_{\mathcal{L}_{\mathrm{NLL}}(\theta)}.
\end{aligned}
$$

这里的 $1/N$ 只把总损失改为平均损失，不改变最优参数。

**接下来连接 MLE 与 KL。** 对固定的 $\theta$，如果负对数似然可积，根据大数定律，

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
\xrightarrow[N\to\infty]{\mathrm{a.s.}}
\mathbb E_{x\sim p_{\mathrm{data}}}[-\log p_\theta(x)]
=
H(p_{\mathrm{data}},p_\theta).
$$

而交叉熵满足

$$
H(p_{\mathrm{data}},p_\theta)
=
H(p_{\mathrm{data}})
+
D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta).
$$

因此，**样本 NLL 是总体交叉熵的经验估计，而总体交叉熵与 Forward KL 只差一个不依赖 $\theta$ 的常数**。这才是 MLE 与最小化 Forward KL 的关系：它们的总体优化目标等价，但有限样本下的最优参数未必等于总体最优参数。

**分类任务中的交叉熵也是同一个形式。** 对带标签样本 $(x^{(i)},y^{(i)})$，模型预测条件概率 $p_\theta(y\mid x)$。将观测标签 $y^{(i)}$ 表示为 one-hot 向量，单个样本的交叉熵为

$$
-\sum_{k=1}^{K}\mathbf{1}[y^{(i)}=k]
\log p_\theta(k\mid x^{(i)})
=
-\log p_\theta(y^{(i)}\mid x^{(i)}).
$$

这里 one-hot 表示的是**观测标签**，并不要求真实条件分布 $p_{\mathrm{data}}(y\mid x)$ 本身一定是 one-hot。

## 4. Forward KL 与 Reverse KL

在比较两种 KL 方向前，先固定约定：$P$ 是**目标分布**，$Q_\theta$ 是**可训练模型**。

$$
\begin{aligned}
\text{Forward KL:}\quad&
D_{\mathrm{KL}}(P\|Q_\theta)
=\mathbb E_{x\sim P}
\log\frac{P(x)}{Q_\theta(x)},\\
\text{Reverse KL:}\quad&
D_{\mathrm{KL}}(Q_\theta\|P)
=\mathbb E_{x\sim Q_\theta}
\log\frac{Q_\theta(x)}{P(x)}.
\end{aligned}
$$

### 4.1 为什么优化方向会影响结果？

Forward KL 在 $P$ 下计算期望。如果 $P$ 在某一区域具有正概率，而 $Q_\theta$ 在该区域为零，那么 Forward KL 为无穷大；因此它严厉惩罚模型**遗漏目标分布的支撑区域**。

Reverse KL 则在 $Q_\theta$ 下计算期望。如果模型在目标 $P$ 为零的区域放置概率质量，那么 Reverse KL 为无穷大；因此它严厉惩罚模型**把概率分配给目标不支持的区域**。

两种惩罚的不对称性，是理解多模态分布拟合行为的起点。

### 4.2 为什么经常说 mode-covering 与 mode-seeking？

考虑目标 $P$ 是由两个相距较远的模态构成的混合分布，而模型 $Q_\theta$ 被限制为**单个高斯分布**。Forward KL 的优化等价于最小化

$$
-\mathbb E_{x\sim P}\log\mathcal N(x;\mu,\sigma^2).
$$

在 $P$ 具有有限方差时，将高斯密度展开：

$$
-\mathbb E_P\log\mathcal N(x;\mu,\sigma^2)
=
\frac12\log(2\pi\sigma^2)
+\frac{\mathbb E_P[(x-\mu)^2]}{2\sigma^2}.
$$

对 $\mu,\sigma^2$ 求极小值可得

$$
\mu^*=\mathbb E_P[x],
\qquad
(\sigma^2)^*=\operatorname{Var}_P(x).
$$

这说明 Forward KL 下的单高斯会**匹配目标分布的均值和方差**；当两个模态相距很远时，它可能用较大的方差同时覆盖两个模态。

Reverse KL 不具有上述矩匹配结论。在模态分离明显且模型只能描述一个模态时，它通常倾向集中在其中一个模态附近，以避免将大量概率放在目标密度很小的区域，这被称为 *mode-seeking*。

这里的 mode-covering/mode-seeking 是**受限模型族下的典型现象，不是无条件的定理**。如果模型能够表示 $P$，两种方向的全局最优都是 $Q_\theta=P$。

## 5. KL 的链式法则

### 5.1 两个随机变量

设 $P(X,Y)$ 和 $Q(X,Y)$ 是两个联合分布。利用条件概率分解，

$$
P(x,y)=P(x)P(y\mid x),
\qquad
Q(x,y)=Q(x)Q(y\mid x).
$$

因此，对数概率比可以拆成两项：

$$
\log\frac{P(x,y)}{Q(x,y)}
=
\log\frac{P(x)}{Q(x)}
+
\log\frac{P(y\mid x)}{Q(y\mid x)}.
$$

在 $P(x,y)$ 下取期望，即得到 **KL 的链式法则**：

$$
\begin{aligned}
&D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))\\
&=D_{\mathrm{KL}}(P(X)\|Q(X))\\
&\quad+
\mathbb E_{x\sim P(X)}
\left[
D_{\mathrm{KL}}\left(
P(Y\mid x)\|Q(Y\mid x)
\right)
\right].
\end{aligned}
$$

它是精确的概率恒等式，而不只是近似。特别注意：条件 KL 外层的期望由**第一个分布 $P(X)$**决定。

### 5.2 边缘化与数据处理不等式

由条件 KL 非负，立刻得到

$$
D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))
\geq D_{\mathrm{KL}}(P(X)\|Q(X)).
$$

也就是说，对两个联合分布都边缘化掉 $Y$，KL 不会增加。

更一般地，设两个输入分布 $P_X,Q_X$ 通过**相同**的随机转移 $K(y\mid x)$ 分别得到 $P_Y,Q_Y$，则

$$
D_{\mathrm{KL}}(P_Y\|Q_Y)
\leq D_{\mathrm{KL}}(P_X\|Q_X).
$$

这称为数据处理不等式（data processing inequality）。一个证明方法是对联合分布 $P_X(x)K(y\mid x)$、$Q_X(x)K(y\mid x)$ 使用链式法则：条件转移相同，联合 KL 就等于输入 KL；再边缘化掉 $X$ 即得到结论。

### 5.3 多步轨迹的 KL

将链式法则重复应用于 $x_{0:T}=(x_0,\ldots,x_T)$：

$$
\begin{aligned}
&D_{\mathrm{KL}}(P(x_{0:T})\|Q(x_{0:T}))\\
&=D_{\mathrm{KL}}(P(x_0)\|Q(x_0))\\
&\quad+\sum_{t=1}^{T}
\mathbb E_{x_{0:t-1}\sim P}
\left[
D_{\mathrm{KL}}\left(
P(x_t\mid x_{0:t-1})
\|Q(x_t\mid x_{0:t-1})
\right)
\right].
\end{aligned}
$$

这是后面分析**扩散轨迹**和**训练状态分布**的基础。每一步的条件 KL 必须在正确的历史状态分布下取期望，不能任意将 $x_{0:t-1}\sim P$ 改成 $x_{0:t-1}\sim Q$。

## 6. KL 与变分推断：ELBO

### 6.1 为什么引入潜变量分布？

考虑潜变量生成模型 $p_\theta(x,z)$。我们能计算联合密度，但计算边缘似然

$$
p_\theta(x)=\int p_\theta(x,z)\,dz
$$

可能十分困难。于是引入辅助后验 $q_\phi(z\mid x)$，构造可优化的对数似然下界。

为简化记号，以下假设 $p_\theta(x)>0$，并且所出现的 KL 与期望有意义。

### 6.2 从精确恒等式推导 ELBO

由 Bayes 公式，

$$
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}.
$$

从辅助后验与真实后验的 KL 出发：

$$
\begin{aligned}
&D_{\mathrm{KL}}\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right)\\
&=\mathbb E_{z\sim q_\phi(z\mid x)}
\left[
\log q_\phi(z\mid x)-\log p_\theta(z\mid x)
\right]\\
&=\log p_\theta(x)
-\mathbb E_{z\sim q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
\end{aligned}
$$

定义证据下界（evidence lower bound, ELBO）：

$$
\mathcal E(\theta,\phi;x)
:=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
$$

于是得到

$$
\log p_\theta(x)
=
\mathcal E(\theta,\phi;x)
+
D_{\mathrm{KL}}\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right).
$$

这是**精确恒等式**。由于 KL 非负，

$$
\mathcal E(\theta,\phi;x)\leq\log p_\theta(x).
$$

更进一步，**下界松弛多少是明确可量化的**：间隙恰好等于 $q_\phi(z\mid x)$ 与真实后验 $p_\theta(z\mid x)$ 的 KL。选择辅助分布的关键并不只是它“容易采样”，还包括它能否近似真实后验。

### 6.3 为什么 VAE 的 ELBO 里有 KL 正则项？

如果联合分布分解为

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z),
$$

那么 ELBO 还能写成

$$
\begin{aligned}
\mathcal E(\theta,\phi;x)
&=\mathbb E_{q_\phi(z\mid x)}
\left[\log p_\theta(x\mid z)\right]\\
&\quad-
D_{\mathrm{KL}}\left(
q_\phi(z\mid x)\|p(z)
\right).
\end{aligned}
$$

这就是常说的“重建项减去 KL 正则项”。但这里的**先验 KL** 与上一小节衡量松弛量的**后验 KL** 是不同的对象：一个比较 $q_\phi$ 与 $p(z)$，另一个比较 $q_\phi$ 与 $p_\theta(z\mid x)$。

## 7. 扩散模型中的 KL：为什么损失能够分解到每个时间步？

### 7.1 前向与反向轨迹

标准离散时间扩散模型先定义一个**固定的前向 Markov 链**：

$$
q(x_{1:T}\mid x_0)
=\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
$$

模型的联合生成分布则定义为

$$
p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).
$$

其中 $x_0$ 是数据，$x_T$ 通常接近一个简单的先验；$q$ 是前向噪声机制，$p_\theta$ 是需要学习的反向过程。

### 7.2 先建立整条轨迹的变分目标

固定一个观测 $x_0$，以 $q(x_{1:T}\mid x_0)$ 作为第 6 节的辅助后验：

$$
\begin{aligned}
-\log p_\theta(x_0)
&\leq
\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\log\frac{q(x_{1:T}\mid x_0)}
{p_\theta(x_{0:T})}
\right]\\
&=:\mathcal L_{\mathrm{VLB}}(x_0).
\end{aligned}
$$

需要区分符号：**ELBO 是 $\log p_\theta(x_0)$ 的下界，而 $\mathcal L_{\mathrm{VLB}}$ 是其负号，即负对数似然的上界**。

### 7.3 再利用 Markov 结构分解

尽管 $q$ 是按正向时间定义的，它也可以重新写成

$$
q(x_{1:T}\mid x_0)
=
q(x_T\mid x_0)
\prod_{t=2}^{T}q(x_{t-1}\mid x_t,x_0).
$$

这里出现的后验并非额外假设，它可以由前向转移直接得到：

$$
q(x_{t-1}\mid x_t,x_0)
=
\frac{
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)
}{
q(x_t\mid x_0)
}.
$$

将前向、反向轨迹的分解代入 $\mathcal L_{\mathrm{VLB}}$，把乘积变为对数的求和，再依 KL 的定义整理：

$$
\begin{aligned}
\mathcal L_{\mathrm{VLB}}(x_0)
&=D_{\mathrm{KL}}\left(
q(x_T\mid x_0)\|p(x_T)
\right)\\
&\quad+\sum_{t=2}^{T}
\mathbb E_{x_t\sim q(x_t\mid x_0)}
\left[
D_{\mathrm{KL}}\left(
q(x_{t-1}\mid x_t,x_0)
\|p_\theta(x_{t-1}\mid x_t)
\right)
\right]\\
&\quad-\mathbb E_{x_1\sim q(x_1\mid x_0)}
\left[\log p_\theta(x_0\mid x_1)\right].
\end{aligned}
$$

三项分别对应**先验匹配、逐步反向转移拟合、最后一步重建**。这条推导的逻辑是：**先定义联合生成模型并构造整体变分界，再由轨迹的概率分解得到逐步 KL**，不是预先人为选择逐步 KL 作为训练目标。

对于 masked diffusion，同样需要指定具体遮罩转移和反向参数化，才能判断哪些 KL 项可以化为加权交叉熵。**交叉熵形状的 loss 并不自动意味着它是某个 ELBO。**

## 8. KL 与可编辑生成过程

### 8.1 最终样本与生成轨迹不是同一个随机变量

对于能够反复修改状态的模型，设 $\tau=(s_0,a_0,\ldots,s_T)$ 表示完整生成轨迹，最终输出由 $x=f(\tau)$ 给出。轨迹分布 $p_\theta(\tau)$ 诱导出输出分布

$$
p_\theta(x)
=
\sum_{\tau:f(\tau)=x}p_\theta(\tau),
$$

这里写的是离散轨迹情形；连续情形将求和换为相应积分。

由于 $x=f(\tau)$ 是对轨迹的确定映射，数据处理不等式说明

$$
D_{\mathrm{KL}}(P(\tau)\|Q(\tau))
\geq D_{\mathrm{KL}}(P(x)\|Q(x)).
$$

因此，**要求两个模型的完整轨迹匹配，通常比仅要求最终输出分布匹配更强**。不同轨迹完全可能得到同一个最终结果。

### 8.2 没有预先设定的前向过程，仍能构造 ELBO 吗？

如果模型定义了规范化的轨迹概率 $p_\theta(\tau)$，答案是可以。给定目标输出 $x$，引入一个**仅支持 $f(\tau)=x$ 的辅助轨迹分布** $r_\phi(\tau\mid x)$。由第 6 节同样的后验 KL 恒等式，

$$
\begin{aligned}
\log p_\theta(x)
&=\mathbb E_{\tau\sim r_\phi(\tau\mid x)}
\left[\log p_\theta(\tau)-\log r_\phi(\tau\mid x)\right]\\
&\quad+
D_{\mathrm{KL}}\left(
r_\phi(\tau\mid x)\|p_\theta(\tau\mid x)
\right).
\end{aligned}
$$

第一项因此是最终对数似然的下界。**构造一个形式上的 ELBO 不需要预先存在扩散式前向加噪过程**；真正的问题在于：如何选择可计算且足够紧的 $r_\phi$，以及如何评价轨迹空间过多的自由度。

### 8.3 训练状态分布与目标函数需要分别说明

在多步模型中，训练可以使用某个参考过程提供的状态，也可以使用模型当前策略生成的状态。两者对应不同的状态分布，因此即使使用相同的逐步监督损失，优化目标也可能不同。

采用 on-policy 状态采样能够减少**训练状态与实际推理状态的分布差异**，但这并不自动意味着相应损失就是最终输出负对数似然的无偏估计，也不保证它具有紧致的变分界。

分析一项可编辑生成训练方法时，应把三个问题分开：

1. **概率模型：**最终输出 $p_\theta(x)$ 如何由轨迹定义？
2. **目标函数：**优化的是精确对数似然、明确的变分下界，还是逐步替代损失？
3. **采样分布：**损失的期望在哪一种中间状态分布上计算？

只有把这些对象明确区分，才能严格讨论训练—推理一致性与目标函数的关系。

---

与本文相关的 KL 严格性质和进一步证明可参阅 [Properties of KL Divergence](https://egnarochan.github.io/Egnaro/properties_of_kl_divergence.html)。
