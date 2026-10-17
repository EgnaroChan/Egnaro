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

$$
\begin{aligned}
D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)
&=\mathbb E_{x\sim p_{\mathrm{data}}}
\left[\log\frac{p_{\mathrm{data}}(x)}{p_\theta(x)}\right]\\
&=\mathbb E_{x\sim p_{\mathrm{data}}}\log p_{\mathrm{data}}(x)
-\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x)\\
&=-H(p_{\mathrm{data}})
-\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x).
\end{aligned}
$$

这里 $p_{\mathrm{data}}$ 是固定的真实分布，不随参数 $\theta$ 改变，因此 $H(p_{\mathrm{data}})$ 对 $\theta$ 来说是常数。最小化 KL 时，可以忽略这个常数项：

$$
\begin{aligned}
\arg\min_\theta D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)
&=\arg\min_\theta\left[
-\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x)
\right]\\
&=\arg\max_\theta
\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x).
\end{aligned}
$$

也就是说，**最小化真实分布到模型分布的 Forward KL，等价于最大化模型在真实数据上的平均对数概率**。这里的期望始终是对 $p_{\mathrm{data}}$ 计算的，而不是对模型分布 $p_\theta$ 计算。

前面的等式给出了一个理想目标：让模型在真实数据分布下的平均对数概率尽可能大。但我们并不知道 $p_{\mathrm{data}}$ 的具体形式，因此无法直接计算这个期望。实际拥有的是从真实分布中独立抽取的 $N$ 个训练样本：

$$
\mathcal D=\{x^{(1)},x^{(2)},\ldots,x^{(N)}\}.
$$

**第一步：用样本平均代替总体期望。** 既然每个训练样本都来自 $p_{\mathrm{data}}$，就可以用模型在这些样本上的平均对数概率，估计它在真实分布下的期望对数概率：

$$
\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x)
\;\approx\;
\frac1N\sum_{i=1}^{N}\log p_\theta(x^{(i)}).
$$

于是，训练时最大化右侧即可。通常把它乘以 $-1$，写成最小化的形式，这就得到**平均负对数似然（NLL）损失**：

$$
\mathcal L_{\mathrm{NLL}}(\theta)
=-\frac1N\sum_{i=1}^{N}\log p_\theta(x^{(i)}).
$$

对某个训练样本，模型给它的概率（连续情形下为概率密度）越高，$-\log p_\theta(x^{(i)})$ 就越小。因此最小化 NLL，就是让模型尽可能提高训练数据的对数概率。

**第二步：为什么这叫最大似然估计（MLE）？** 如果训练样本相互独立，模型在参数 $\theta$ 下给整批数据的联合似然是

$$
p_\theta(\mathcal D)
=\prod_{i=1}^{N}p_\theta(x^{(i)}).
$$

所谓**最大似然估计**，就是选择使这个乘积最大的参数。由于对数函数严格递增，最大化乘积等价于最大化对数之和；再乘以负数、除以 $N$，就变成最小化刚才的 NLL：

$$
\begin{aligned}
\hat\theta_{\mathrm{MLE}}
&=\arg\max_\theta\prod_{i=1}^{N}p_\theta(x^{(i)})\\
&=\arg\max_\theta\sum_{i=1}^{N}\log p_\theta(x^{(i)})\\
&=\arg\min_\theta\mathcal L_{\mathrm{NLL}}(\theta).
\end{aligned}
$$

因此，**最大似然估计（MLE）和最小化训练集 NLL 是完全等价的两种写法**。

**最后，把它与 Forward KL 联系起来：** 理想情况下，我们希望最小化 $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)$；但真实分布未知，所以实际训练通过样本 NLL 来估计相应的总体优化目标。两者**不是在有限数据集上严格相等**：有限样本的平均值通常不等于真实期望，会存在抽样误差；在常见的独立同分布及可积条件下，随着样本数增多，对固定模型参数的样本平均会趋近真实期望。

在分类任务中，模型预测的是条件概率 $p_\theta(y\mid x)$。用同样的思路，对每个带标签样本计算 $-\log p_\theta(y\mid x)$，得到常见的分类交叉熵损失。

## 4. Forward KL 与 Reverse KL

如果交换两个分布的位置，优化目标是否不变？一般不会，因为**KL 的第一个参数决定了取期望的分布**。

令 $P$ 表示目标分布，$Q_\theta$ 表示模型。两种方向分别是

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q_\theta)
&=\mathbb E_{x\sim P}\left[\log P(x)-\log Q_\theta(x)\right],\\
D_{\mathrm{KL}}(Q_\theta\|P)
&=\mathbb E_{x\sim Q_\theta}\left[\log Q_\theta(x)-\log P(x)\right].
\end{aligned}
$$

### 4.1 两个方向分别惩罚什么？

Forward KL 在目标分布 $P$ 下取期望，因此要求模型在真实数据经常出现的地方分配足够的概率。如果 $P(x)>0$、$Q_\theta(x)=0$，那么 $D_{\mathrm{KL}}(P\|Q_\theta)=+\infty$。

Reverse KL 则在模型分布 $Q_\theta$ 下取期望。如果 $Q_\theta(x)>0$、$P(x)=0$，那么 $D_{\mathrm{KL}}(Q_\theta\|P)=+\infty$。它直接惩罚模型在目标分布不支持的地方生成数据。

这些是关于**支撑集**的严格结论。对于高斯等处处为正的密度，两种优化行为的区别需要结合模型的表达能力具体分析。

### 4.2 单高斯拟合双峰分布

考虑一个双峰目标分布

$$
P(x)=\frac12\mathcal N(x;-a,s^2)
+\frac12\mathcal N(x;a,s^2),\qquad a>0,
$$

但模型只能表示单个高斯 $Q_{\mu,v}(x)=\mathcal N(x;\mu,v)$，其中 $v>0$。

从 Forward KL 出发，将高斯的对数密度代入，

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q_{\mu,v})
&=-H(P)-\mathbb E_{x\sim P}\log Q_{\mu,v}(x)\\
&=-H(P)+\frac12\log(2\pi v)
+\frac{\mathbb E_P[(x-\mu)^2]}{2v}.
\end{aligned}
$$

由于 $\mathbb E_P[x]=0$、$\operatorname{Var}_P(x)=a^2+s^2$，以及

$$
\mathbb E_P[(x-\mu)^2]
=\operatorname{Var}_P(x)+(\mathbb E_P[x]-\mu)^2,
$$

可以得到

$$
D_{\mathrm{KL}}(P\|Q_{\mu,v})
=\text{const}+\frac12\log v
+\frac{a^2+s^2+\mu^2}{2v},
$$

其中常数项与 $\mu,v$ 无关。分别对 $\mu$ 和 $v$ 求极小值：

$$
\mu^*=0,\qquad v^*=a^2+s^2.
$$

当两峰相隔很远时，单高斯会取位于两峰之间、方差较大的解。这是 Forward KL 常见的 *mode-covering* 行为。

而 Reverse KL 的目标是

$$
D_{\mathrm{KL}}(Q_{\mu,v}\|P)
=\mathbb E_{Q_{\mu,v}}\log Q_{\mu,v}(x)
-\mathbb E_{Q_{\mu,v}}\log P(x).
$$

若模型把较多概率分给目标密度很低的两峰之间的区域，第二项就会带来惩罚。因此，在峰间距离很大且只能使用单峰模型时，较优解往往偏向其中一个峰，表现出 *mode-seeking*。这里并不存在与 Forward KL 相同的矩匹配结论。

**mode-covering 和 mode-seeking 是模型族受限时的典型现象，不是关于 KL 的无条件定理。** 当 $Q_\theta$ 能够精确等于 $P$ 时，两种方向都在 $Q_\theta=P$ 处取得最小值 $0$。

## 5. KL 的链式法则与数据处理不等式

对于联合分布，KL 中的概率比可以先按条件概率分解，再对各部分分别取期望。这是后续推导 ELBO 与轨迹目标的重要工具。

### 5.1 链式法则如何得到？

设 $P(x,y)$ 和 $Q(x,y)$ 为两个联合分布，以下假设相关概率比有定义。由条件概率公式，

$$
P(x,y)=P(x)P(y\mid x),\qquad
Q(x,y)=Q(x)Q(y\mid x).
$$

代入对数比，并利用乘积的对数等于对数之和：

$$
\begin{aligned}
\log\frac{P(x,y)}{Q(x,y)}
&=\log\frac{P(x)P(y\mid x)}{Q(x)Q(y\mid x)}\\
&=\log\frac{P(x)}{Q(x)}
+\log\frac{P(y\mid x)}{Q(y\mid x)}.
\end{aligned}
$$

对 $P(x,y)$ 取期望。第一项只依赖 $x$，所以对 $y$ 求和后剩下 $P(x)$；第二项先在 $P(y\mid x)$ 下取期望，然后在 $P(x)$ 下取期望。因此

$$
\begin{aligned}
&D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))\\
&=D_{\mathrm{KL}}(P(X)\|Q(X))\\
&\quad+\mathbb E_{x\sim P(X)}
\left[D_{\mathrm{KL}}\bigl(P(Y\mid x)\|Q(Y\mid x)\bigr)\right].
\end{aligned}
$$

这是**精确等式**。特别注意：条件 KL 外的期望由第一个分布 $P(X)$ 决定，不能任意换成 $Q(X)$。

### 5.2 为什么数据处理不会增加 KL？

链式法则右侧的条件 KL 非负，所以

$$
D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))
\geq D_{\mathrm{KL}}(P(X)\|Q(X)).
$$

即对两个联合分布都边缘化掉 $Y$，KL 不会增加。

更一般地，假设 $P_X$、$Q_X$ 通过**同一个**随机转移 $K(y\mid x)$ 分别得到 $P_Y$、$Q_Y$。构造联合分布

$$
P_{XY}(x,y)=P_X(x)K(y\mid x),\qquad
Q_{XY}(x,y)=Q_X(x)K(y\mid x).
$$

由于两者的条件转移相同，链式法则给出

$$
D_{\mathrm{KL}}(P_{XY}\|Q_{XY})
=D_{\mathrm{KL}}(P_X\|Q_X).
$$

再通过边缘化掉 $X$，

$$
D_{\mathrm{KL}}(P_Y\|Q_Y)
\leq D_{\mathrm{KL}}(P_{XY}\|Q_{XY}).
$$

合并两式，得到**数据处理不等式（data processing inequality）**：

$$
D_{\mathrm{KL}}(P_Y\|Q_Y)
\leq D_{\mathrm{KL}}(P_X\|Q_X).
$$

这个结论要求两个输入分布使用相同的随机映射；如果映射也发生变化，不能直接这样比较。

### 5.3 多步随机过程

将链式法则反复用于轨迹 $x_{0:T}=(x_0,\ldots,x_T)$，有

$$
\begin{aligned}
&D_{\mathrm{KL}}(P(x_{0:T})\|Q(x_{0:T}))\\
&=D_{\mathrm{KL}}(P(x_0)\|Q(x_0))\\
&\quad+\sum_{t=1}^{T}
\mathbb E_{x_{0:t-1}\sim P}
\left[D_{\mathrm{KL}}\bigl(
P(x_t\mid x_{0:t-1})
\|Q(x_t\mid x_{0:t-1})
\bigr)\right].
\end{aligned}
$$

一般过程的条件分布可能依赖**全部历史**；只有满足 Markov 性时才能把历史缩减为上一个状态。

这里不能遗漏两个关键条件：每步比较的是**同一个历史条件下**的转移分布，而且每步的 KL 在**第一个过程实际产生的历史分布**下取期望。擅自把这个期望替换成 $Q$ 的历史状态分布，通常就不再是原来的轨迹 KL。

## 6. KL 与证据下界（ELBO）

KL 不仅能作为优化目标，还能帮助我们推导生成模型的变分下界。先讨论一般潜变量模型，再将结果应用到扩散模型。

### 6.1 从边缘似然构造下界

设生成模型有联合分布 $p_\theta(x,z)$，其中 $x$ 是观测数据，$z$ 是潜变量。我们希望最大化

$$
\log p_\theta(x)=\log\int p_\theta(x,z)\,dz,
$$

但边缘积分可能难以计算。引入辅助分布 $q_\phi(z\mid x)$。为保证以下等式成立，假设对固定 $x$，在 $p_\theta(x,z)>0$ 的地方 $q_\phi(z\mid x)>0$，且相关期望有定义。将 $q_\phi$ 乘入积分再除去：

$$
\begin{aligned}
\log p_\theta(x)
&=\log\int q_\phi(z\mid x)
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz\\
&=\log\mathbb E_{z\sim q_\phi(z\mid x)}
\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right].
\end{aligned}
$$

$\log$ 是凹函数，由 Jensen 不等式，

$$
\log p_\theta(x)
\geq\mathbb E_{q_\phi(z\mid x)}
\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right].
$$

将右侧定义为**证据下界（evidence lower bound, ELBO）**：

$$
\mathcal E(\theta,\phi;x)
:=\mathbb E_{q_\phi(z\mid x)}
\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right].
$$

所以它是对数似然的下界，原因是 Jensen 不等式，而不是因为我们人为添加了某个 KL 正则项。

### 6.2 下界松弛了多少？

由 Bayes 公式

$$
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}.
$$

从辅助后验和模型真实后验之间的 KL 出发：

$$
\begin{aligned}
&D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\|p_\theta(z\mid x)\bigr)\\
&=\mathbb E_{q_\phi}
\left[\log q_\phi(z\mid x)-\log p_\theta(x,z)+\log p_\theta(x)\right]\\
&=\log p_\theta(x)-\mathcal E(\theta,\phi;x).
\end{aligned}
$$

于是得到精确恒等式

$$
\log p_\theta(x)
=\mathcal E(\theta,\phi;x)
+D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\|p_\theta(z\mid x)\bigr).
$$

这说明：**ELBO 与真实对数似然之间的间隙，恰好是辅助后验与真实后验的 KL**。当这两个后验相等时，下界取等。

### 6.3 为什么 VAE 的 ELBO 含有 KL？

若模型的联合分布写为

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z),
$$

代入 ELBO 的定义，整理得到

$$
\begin{aligned}
\mathcal E(\theta,\phi;x)
&=\mathbb E_{q_\phi(z\mid x)}\log p_\theta(x\mid z)\\
&\quad-D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\|p(z)\bigr).
\end{aligned}
$$

这就是 VAE 中的重建项和先验 KL 项。需要区分两种不同的 KL：**$D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$ 是优化目标的一部分；$D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_\theta(z\mid x))$ 衡量下界的松弛程度。** 前者不是后者。

## 7. 扩散模型中的逐时间步 KL

扩散模型的潜变量不是单个 $z$，而是整条加噪轨迹 $x_{1:T}$。为什么它的训练目标经常包含逐时间步 KL？原因在于轨迹分布的概率分解。

### 7.1 前向加噪与反向生成

固定的前向 Markov 链满足

$$
q(x_{1:T}\mid x_0)
=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),
$$

反向生成模型则定义为

$$
p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).
$$

$x_0$ 是数据，$p(x_T)$ 是先验。$q$ 的转移已知，$p_\theta$ 的反向转移需要学习。目标是提高最终数据的边缘似然 $p_\theta(x_0)$，而不是为某一条特定噪声轨迹分配最大概率。

### 7.2 先建立整个轨迹的变分目标

对固定的 $x_0$，将 $q(x_{1:T}\mid x_0)$ 作为第 6 节的辅助分布，有

$$
\log p_\theta(x_0)
\geq\mathbb E_{q(x_{1:T}\mid x_0)}
\left[\log p_\theta(x_{0:T})-\log q(x_{1:T}\mid x_0)\right].
$$

对不等式取负号，得到负对数似然的**上界**

$$
-\log p_\theta(x_0)
\leq\mathcal L_{\mathrm{VLB}}(x_0)
:=\mathbb E_q\left[
\log\frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}
\right].
$$

注意：ELBO 是 $\log p_\theta(x_0)$ 的下界；其负值 $\mathcal L_{\mathrm{VLB}}$ 是 $-\log p_\theta(x_0)$ 的上界。

### 7.3 把整体目标分解为逐步 KL

利用 Bayes 公式，同一个前向轨迹分布可以改写为

$$
q(x_{1:T}\mid x_0)
=q(x_T\mid x_0)\prod_{t=2}^{T}q(x_{t-1}\mid x_t,x_0),
$$

其中

$$
q(x_{t-1}\mid x_t,x_0)
=\frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}
{q(x_t\mid x_0)}
$$

在分母为正时成立。把这个表达式和反向模型代入 $\mathcal L_{\mathrm{VLB}}$，对乘积取对数：

$$
\begin{aligned}
\log\frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}
&=\log\frac{q(x_T\mid x_0)}{p(x_T)}\\
&\quad+\sum_{t=2}^{T}
\log\frac{q(x_{t-1}\mid x_t,x_0)}
{p_\theta(x_{t-1}\mid x_t)}\\
&\quad-\log p_\theta(x_0\mid x_1).
\end{aligned}
$$

最后对 $q(x_{1:T}\mid x_0)$ 取期望。每个求和项先对 $q(x_{t-1}\mid x_t,x_0)$ 求期望，得到条件 KL，再对 $q(x_t\mid x_0)$ 求期望：

$$
\begin{aligned}
\mathcal L_{\mathrm{VLB}}(x_0)
&=D_{\mathrm{KL}}\bigl(q(x_T\mid x_0)\|p(x_T)\bigr)\\
&\quad+\sum_{t=2}^{T}\mathbb E_{x_t\sim q(x_t\mid x_0)}
\left[D_{\mathrm{KL}}\bigl(
q(x_{t-1}\mid x_t,x_0)
\|p_\theta(x_{t-1}\mid x_t)
\bigr)\right]\\
&\quad-\mathbb E_{x_1\sim q(x_1\mid x_0)}
\log p_\theta(x_0\mid x_1).
\end{aligned}
$$

三项分别是末端先验匹配、逐步反向转移拟合和最后一步重建。**先定义完整概率模型并构造 ELBO，随后由概率分解得到逐步 KL；并非一开始就人为指定逐步 KL 作为训练目标。**

### 7.4 Masked diffusion 为什么常使用交叉熵？

考虑一个简单的吸收式遮罩过程：每个位置独立加噪，一旦某个 token 变为特殊遮罩符号 $m$，以后就一直保持遮罩。令 $\alpha_t$ 表示一个位置在时刻 $t$ 仍为原始 token 的概率，并设 $\alpha_0=1$、$\alpha_{t-1}\geq\alpha_t$。

对于在时刻 $t$ 被遮罩的位置 $i$，它在时刻 $t-1$ 要么已经被遮罩，要么刚从原始 token 变为遮罩。后一种情况的条件概率为

$$
c_t
=\Pr(x_{t-1}^i=x_0^i\mid x_t^i=m,x_0)
=\frac{\alpha_{t-1}-\alpha_t}{1-\alpha_t}.
$$

因此真实反向后验是

$$
q(x_{t-1}^i\mid x_t^i=m,x_0)
=(1-c_t)\delta_m+c_t\delta_{x_0^i},
$$

其中 $\delta_u$ 为集中在 $u$ 的点质量。假设反向模型保留这个遮罩概率，只把解遮罩时的真实 token 替换为预测分布 $\hat p_\theta(\cdot\mid x_t)$：

$$
p_\theta(x_{t-1}^i\mid x_t)
=(1-c_t)\delta_m+c_t\hat p_\theta(\cdot\mid x_t).
$$

若 $\hat p_\theta$ 只对非遮罩 token 赋予概率，对应的条件 KL 可以直接计算：

$$
\begin{aligned}
&D_{\mathrm{KL}}\bigl(
q(x_{t-1}^i\mid x_t^i=m,x_0)
\|p_\theta(x_{t-1}^i\mid x_t)
\bigr)\\
&=c_t\log\frac{c_t}{c_t\hat p_\theta(x_0^i\mid x_t)}\\
&=c_t\bigl[-\log\hat p_\theta(x_0^i\mid x_t)\bigr].
\end{aligned}
$$

对于未遮罩的位置，如果真实后验与模型转移都保持该 token，条件 KL 为零。当各位置的反向转移在给定 $x_t$ 时条件独立，整个时间步的条件 KL 就是这些位置级 KL 之和：

$$
D_{\mathrm{KL}}\bigl(
q(x_{t-1}\mid x_t,x_0)
\|p_\theta(x_{t-1}\mid x_t)
\bigr)
=c_t\sum_{i:x_t^i=m}\bigl[-\log\hat p_\theta(x_0^i\mid x_t)\bigr].
$$

进一步在 $q(x_t\mid x_0)$ 下取期望，便得到第 7.3 节 ELBO 中对应时间步的训练项。

这就是**带时间权重的 token 交叉熵**。在这个特定模型中，交叉熵由已知的前向后验和反向参数化推导得到；更换噪声过程或允许反向步骤编辑已有 token 后，需要重新检查是否仍有同样的等式。

## 8. 最终输出分布与生成轨迹

如果生成过程允许反复编辑状态，模型不一定有预先设定的加噪过程。这时仍可使用 KL 和 ELBO，但必须区分**最终输出的概率**与**整条生成轨迹的概率**。

### 8.1 为什么两种 KL 不相同？

设 $\tau=(s_0,a_0,\ldots,s_T)$ 表示完整生成轨迹，最终输出 $x=f(\tau)$。给定模型的轨迹分布 $p_\theta(\tau)$，它诱导出最终输出分布

$$
p_\theta(x)=\sum_{\tau:f(\tau)=x}p_\theta(\tau).
$$

这里用离散记号；连续轨迹情形使用相应积分。不同轨迹可能得到同一个输出，因此生成轨迹分布与最终输出分布并不是同一个对象。

设 $P_\tau,Q_\tau$ 是两种轨迹分布，$P_X,Q_X$ 为相同映射 $f$ 诱导出的输出分布。对联合变量 $(\tau,x)$ 应用第 5 节链式法则，可得

$$
\begin{aligned}
D_{\mathrm{KL}}(P_\tau\|Q_\tau)
&=D_{\mathrm{KL}}(P_X\|Q_X)\\
&\quad+\mathbb E_{x\sim P_X}
\left[D_{\mathrm{KL}}\bigl(
P(\tau\mid x)\|Q(\tau\mid x)
\bigr)\right].
\end{aligned}
$$

在相关条件分布与 KL 有定义时这是精确等式。因此，**轨迹 KL 比最终输出 KL 多出的部分，恰好是在给定同一个最终输出后，两种轨迹条件分布的平均 KL**。用固定参考轨迹监督模型，可能强制它学习某种生成路径，而不仅是最终输出。

### 8.2 没有固定前向过程，也能构造 ELBO 吗？

如果模型定义了规范化的轨迹概率 $p_\theta(\tau)$，最终输出仍由 $x=f(\tau)$ 决定。对于给定的 $x$，引入辅助轨迹分布 $r_\phi(\tau\mid x)$，并要求它只在 $f(\tau)=x$ 的轨迹上取正概率。第 6 节的恒等式给出

$$
\begin{aligned}
\log p_\theta(x)
&=\mathbb E_{\tau\sim r_\phi(\tau\mid x)}
\left[\log p_\theta(\tau)-\log r_\phi(\tau\mid x)\right]\\
&\quad+D_{\mathrm{KL}}\bigl(
r_\phi(\tau\mid x)\|p_\theta(\tau\mid x)
\bigr).
\end{aligned}
$$

因此第一项是最终对数似然的下界，间隙是辅助轨迹后验与真实轨迹后验的 KL。这个构造**不需要预先指定扩散式前向加噪过程**。但它并未自动提供可计算且足够紧的下界；如何评价 $p_\theta(\tau)$、如何构造 $r_\phi$ 仍是实际问题。

### 8.3 训练状态分布与目标函数须分别明确

设 $\ell_\theta(s_t,x)$ 为中间状态 $s_t$ 下针对最终目标 $x$ 的训练损失。如果状态由固定参考过程产生，其目标是

$$
\mathcal J_{\mathrm{ref}}(\theta)
=\mathbb E_{x\sim p_{\mathrm{data}}}
\sum_t\mathbb E_{s_t\sim\nu_t(\cdot\mid x)}
\left[\ell_\theta(s_t,x)\right].
$$

如果状态由当前模型自己生成，记状态分布为 $d_{\theta,t}(s_t\mid x)$，则目标变为

$$
\mathcal J_{\mathrm{on}}(\theta)
=\mathbb E_{x\sim p_{\mathrm{data}}}
\sum_t\mathbb E_{s_t\sim d_{\theta,t}(\cdot\mid x)}
\left[\ell_\theta(s_t,x)\right].
$$

即使逐步损失相同，**取期望的状态分布不同，整体目标一般也不同**。On-policy 状态采样有助于缩小训练与推理时的状态分布差异，但不能自动说明逐步损失等于最终输出的负对数似然，或等于它的某个 ELBO。

此外，$d_{\theta,t}$ 本身依赖 $\theta$，严格计算 $\nabla_\theta\mathcal J_{\mathrm{on}}$ 时也需要考虑采样分布的参数依赖。仅在采样出的状态上对 $\ell_\theta$ 反向传播，通常不是完整目标的梯度。

因此，研究可编辑生成方法时需要分别交代：**最终分布 $p_\theta(x)$ 如何定义；训练目标与最终似然的数学关系是什么；训练损失在哪种中间状态分布下取期望。** 这三个问题不能互相替代。

---

更多 KL 的数学性质参阅 [Properties of KL Divergence](https://egnarochan.github.io/Egnaro/properties_of_kl_divergence.html)。
