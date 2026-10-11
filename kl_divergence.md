# KL Divergence（KL 散度）

> 系统学习笔记：KL 的数学基础、信息论意义、最大似然、变分推断与扩散模型。

## 1. 定义

**Kullback–Leibler divergence（KL 散度）**，也称为**相对熵（Relative Entropy）**，是描述两个概率分布之间差异的量。它不是对称的，因此不是严格意义上的距离。

对于定义在同一离散样本空间 $\mathcal X$ 上的两个概率分布 $P$ 和 $Q$：

$$

D_{\mathrm{KL}}(P\|Q)
=
\sum_{x\in\mathcal X}P(x)\log\frac{P(x)}{Q(x)}

$$

等价的期望形式：

$$

D_{\mathrm{KL}}(P\|Q)
=
\mathbb E_{x\sim P}
\left[\log\frac{P(x)}{Q(x)}\right]

$$

如果 $P$ 和 $Q$ 具有相对于同一测度的概率密度 $p(x)$ 和 $q(x)$，那么：

$$
D_{\mathrm{KL}}(P\|Q)
=
\int p(x)\log\frac{p(x)}{q(x)}\,dx.
$$

这里通常使用**自然对数**，因此 KL 的单位为 **nat**（如果使用 $\log_2$，单位为 bit）。

**关于零概率的约定：**

- 当 $P(x)=0$ 时，对应项按 $0\log(0/q)=0$ 处理。
- 如果存在 $P(x)>0$ 而 $Q(x)=0$ 的位置，则 $D_{\mathrm{KL}}(P\|Q)=+\infty$。
- 更一般地，只有当 $P$ 对 $Q$ 绝对连续（记作 $P\ll Q$）时，KL 散度才可能有限。

## 2. 直观理解：多花了多少编码长度？

假设数据实际来自分布 $P$，但我们误以为数据来自 $Q$，并据此设计编码方案。

对一次观测 $x$ 而言，按照信息论中的理想编码长度：

$$
L_P(x)=-\log P(x),\qquad
L_Q(x)=-\log Q(x).
$$

使用 $Q$ 代替 $P$，平均会多付出：

$$
\begin{aligned}
\mathbb E_{x\sim P}[L_Q(x)-L_P(x)]
&=\mathbb E_{x\sim P}\left[-\log Q(x)+\log P(x)\right]\\
&=D_{\mathrm{KL}}(P\|Q).
\end{aligned}
$$

因此：

- **$P$** 是数据实际服从的分布。
- **$Q$** 是我们用来近似或描述数据的分布。
- **$D_{\mathrm{KL}}(P\|Q)$** 是使用 $Q$ 而非 $P$ 时，平均增加的理想编码长度。

这里的“额外编码长度”是一个**平均值**；对于单个 $x$，$\log(P(x)/Q(x))$ 可以是负数。

## 3. 一个简单例子

设：

$$
P=(0.5,0.5),\qquad Q=(0.75,0.25).
$$

代入定义：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=0.5\log\frac{0.5}{0.75}
+0.5\log\frac{0.5}{0.25}\\
&\approx 0.1438.
\end{aligned}
$$

交换顺序：

$$
\begin{aligned}
D_{\mathrm{KL}}(Q\|P)
&=0.75\log\frac{0.75}{0.5}
+0.25\log\frac{0.25}{0.5}\\
&\approx 0.1308.
\end{aligned}
$$

由此可见：

$$
D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P).
$$

**关键点：**两者比较的是同一对分布，但**取期望的分布不同**，因此数值一般不同。

## 4. 基本性质

### 4.1 非负性

$$
D_{\mathrm{KL}}(P\|Q)\ge 0
$$

等号成立当且仅当 $P=Q$（概率分布意义下；密度相等是几乎处处的）。

这称为 **Gibbs 不等式**。证明可以使用 Jensen 不等式，也可以使用 $t\log t-t+1\ge0$。

### 4.2 不对称性

一般而言：

$$
D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P).
$$

所以 KL 散度**不满足距离的对称性要求**，也一般不满足三角不等式。

### 4.3 可能为无穷大

如果 $P$ 会产生某种结果，而 $Q$ 将这种结果判为不可能：

$$
P(x)>0,\quad Q(x)=0
\quad\Longrightarrow\quad
D_{\mathrm{KL}}(P\|Q)=+\infty.
$$

这也是理解 KL 优化行为时很重要的一点。

## 5. 为什么 KL 散度非负？——Jensen 不等式

虽然 KL 散度的定义中包含对数比值，但每一项 $P(x)\log(P(x)/Q(x))$ 不一定非负。关键在于**概率的归一化条件**。

函数 $f(u)=-\log u$ 是严格凸函数。根据 Jensen 不等式：

$$
\mathbb{E}[f(Z)]\geq f(\mathbb{E}[Z]).
$$

先假设在所有 $P(x)>0$ 的位置均有 $Q(x)>0$。令 $Z=Q(x)/P(x)$，其中 $x\sim P$，则：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\mathbb{E}_{x\sim P}
\left[-\log\frac{Q(x)}{P(x)}\right]\\
&\geq -\log\left(
\sum_{x:P(x)>0}P(x)\frac{Q(x)}{P(x)}
\right)\\
&=-\log\left(
\sum_{x:P(x)>0}Q(x)
\right)\\
&\geq 0.
\end{aligned}
$$

最后一步成立，是因为 $\sum_{x:P(x)>0}Q(x)\leq 1$。

**何时取等号？** Jensen 不等式取等要求 $Q(x)/P(x)$ 在 $P$ 的支撑集上为常数；第二个不等式取等要求 $Q$ 在该集合以外没有概率质量。再结合两者概率和为 1，可以得到：$D_{\mathrm{KL}}(P\|Q)=0$ 当且仅当 $P=Q$。

若存在 $P(x)>0$ 而 $Q(x)=0$，KL 为 $+\infty$，非负性同样成立。这一结论称为 **Gibbs 不等式**。

## 6. 熵、交叉熵与 KL 散度

### 6.1 熵（Entropy）

分布 $P$ 的 Shannon 熵定义为：

$$
H(P)
=
-\sum_x P(x)\log P(x)
=
\mathbb{E}_{x\sim P}[-\log P(x)].
$$

它表示来自 $P$ 的样本所携带的平均信息量。

### 6.2 交叉熵（Cross-Entropy）

用分布 $Q$ 描述来自 $P$ 的数据时，交叉熵定义为：

$$
H(P,Q)
=
-\sum_x P(x)\log Q(x)
=
\mathbb{E}_{x\sim P}[-\log Q(x)].
$$

注意：即使出现了 $Q$，**期望仍然是对 $P$ 取的**。

### 6.3 KL 和交叉熵的关系

直接展开 KL 定义：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\sum_x P(x)
\left[\log P(x)-\log Q(x)\right]\\
&=-H(P)+H(P,Q).
\end{aligned}
$$

因此：

$$
D_{\mathrm{KL}}(P\|Q)=H(P,Q)-H(P).
$$

当目标分布 $P$ 不随模型参数变化、$Q=Q_\theta$ 可训练时，$H(P)$ 是常数，于是：

$$
\arg\min_\theta D_{\mathrm{KL}}(P\|Q_\theta)
=
\arg\min_\theta H(P,Q_\theta).
$$

**这就是为什么机器学习中可以通过最小化交叉熵来拟合真实分布**：与最小化 Forward KL 相差一个常数项。

## 7. KL 散度与最大似然估计（MLE）

设真实数据分布是 $p_{\mathrm{data}}(x)$，模型分布是 $p_\theta(x)$。若要让模型尽量逼近真实分布，可以考虑：

$$
\min_\theta
D_{\mathrm{KL}}\left(
p_{\mathrm{data}}\|p_\theta
\right).
$$

展开定义：

$$
\begin{aligned}
D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)
&=\mathbb{E}_{p_{\mathrm{data}}}
[\log p_{\mathrm{data}}(x)]\\
&\quad-
\mathbb{E}_{p_{\mathrm{data}}}
[\log p_\theta(x)].
\end{aligned}
$$

第一项与参数 $\theta$ 无关，所以等价于：

$$
\max_\theta
\mathbb{E}_{x\sim p_{\mathrm{data}}}
[\log p_\theta(x)].
$$

真实数据分布通常未知。给定独立样本 $x^{(1)},\ldots,x^{(N)}$，使用经验平均估计期望：

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log p_\theta(x^{(i)}).
$$

最小化这一负对数似然损失（Negative Log-Likelihood，NLL），就是通常的最大似然估计。

**关键点：MLE 与最小化 $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)$ 相联系，真实分布写在 KL 的第一个位置。**

## 8. Forward KL 与 Reverse KL

设 $P$ 是目标分布，$Q_\theta$ 是模型分布。

**Forward KL（正向 KL）**：

$$
D_{\mathrm{KL}}(P\|Q_\theta)
=
\mathbb{E}_{x\sim P}
\left[\log\frac{P(x)}{Q_\theta(x)}\right].
$$

**Reverse KL（反向 KL）**：

$$
D_{\mathrm{KL}}(Q_\theta\|P)
=
\mathbb{E}_{x\sim Q_\theta}
\left[\log\frac{Q_\theta(x)}{P(x)}\right].
$$

它们的区别不只是符号顺序，**更是取期望的分布不同**。

### 8.1 对零概率的惩罚

- 如果 $P(x)>0$ 而 $Q_\theta(x)=0$，Forward KL 为无穷大，严厉惩罚模型**漏掉真实分布所支持的事件**。
- 如果 $Q_\theta(x)>0$ 而 $P(x)=0$，Reverse KL 为无穷大，严厉惩罚模型**在目标不支持的位置分配质量**。

### 8.2 Mode-covering 与 Mode-seeking

当目标分布存在多个模态，而模型分布受限（例如只能表示近似单峰的分布）时，常常观察到：

- **Forward KL 更偏向 mode-covering**：为了不遗漏重要模态，可能覆盖多个模态。
- **Reverse KL 更偏向 mode-seeking**：为了避免低目标概率的区域，可能集中到其中一个模态。

这是一种在**受限模型族**中常见的优化倾向，不是无条件成立的数学定理。如果模型能够精确表示 $P$，两种 KL 都在 $Q_\theta=P$ 时取得全局最优值 0。

一个极端的离散例子：设 $P=(0.49,0.49,0.02)$，模型只能选择某一个固定输出 $Q=\delta_i$。此时所有可选 $Q$ 的 Forward KL 都是无穷大；而

$$
D_{\mathrm{KL}}(\delta_i\|P)=-\log P(i),
$$

因此 Reverse KL 的最优选择是概率最大的前两个类别之一。这个例子说明了 Reverse KL 可能出现的 mode-seeking 行为，也说明对 Forward KL 作 mode-covering 类比时需要考虑模型族是否有足够的支持集。

## 9. KL 的链式法则（Chain Rule）

这一性质对于序列生成、扩散过程和整条生成轨迹的分析尤其重要。

### 9.1 两个随机变量

设两个联合分布分别为 $P(X,Y)$ 和 $Q(X,Y)$。利用概率分解：

$$
P(x,y)=P(x)P(y\mid x),
\qquad
Q(x,y)=Q(x)Q(y\mid x).
$$

代入 KL 的定义：

$$
\begin{aligned}
&D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))\\
&=\mathbb{E}_{(x,y)\sim P}
\left[
\log\frac{P(x)}{Q(x)}
+
\log\frac{P(y\mid x)}{Q(y\mid x)}
\right]\\
&=D_{\mathrm{KL}}(P(X)\|Q(X))\\
&\quad+
\mathbb{E}_{x\sim P(X)}
\left[
D_{\mathrm{KL}}\left(
P(Y\mid x)\|Q(Y\mid x)
\right)
\right].
\end{aligned}
$$

**结论：联合分布的 KL = 边缘分布的 KL + 对第一个分布取期望的条件 KL。**

### 9.2 边缘化不增加 KL

因为任意条件 KL 非负，所以：

$$
D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))
\geq
D_{\mathrm{KL}}(P(X)\|Q(X)).
$$

直观地说：**把两个分布的信息都做相同的边缘化处理，不会使 KL 增大**。这也可以看成数据处理不等式（Data Processing Inequality）的一个特例。

### 9.3 多步轨迹

设两个轨迹分布按同一变量顺序分解：

$$
P(x_{0:T})=
P(x_0)\prod_{t=1}^T P(x_t\mid x_{0:t-1}),
$$

$$
Q(x_{0:T})=
Q(x_0)\prod_{t=1}^T Q(x_t\mid x_{0:t-1}).
$$

反复运用链式法则：

$$
\begin{aligned}
D_{\mathrm{KL}}(P(x_{0:T})\|Q(x_{0:T}))
&=D_{\mathrm{KL}}(P(x_0)\|Q(x_0))\\
&\quad+\sum_{t=1}^T
\mathbb{E}_{x_{0:t-1}\sim P}
\left[
D_{\mathrm{KL}}\left(
P(x_t\mid x_{0:t-1})
\|Q(x_t\mid x_{0:t-1})
\right)
\right].
\end{aligned}
$$

**关键点：逐步 KL 的期望由整个轨迹 KL 的第一个分布 $P$ 决定。**如果训练使用的状态分布与推理时实际到达的状态分布不同，逐步优化与目标轨迹分布之间的关系就必须重新分析。
