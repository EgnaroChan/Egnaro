# KL Divergence（KL 散度）

> 学习笔记 · 第一部分：定义、直觉与基本性质  
> 后续主题：非负性的证明、熵与交叉熵、Forward/Reverse KL、MLE、ELBO 与扩散模型。

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

## 5. 下一步学习

接下来沿着一条主线逐步扩展：

1. **非负性证明：**从 Jensen 不等式推导 $D_{\mathrm{KL}}\ge0$。
2. **信息熵与交叉熵：**推导 $D_{\mathrm{KL}}(P\|Q)=H(P,Q)-H(P)$。
3. **Forward KL / Reverse KL：**理解优化方向对结果的影响。
4. **最大似然估计：**理解为什么拟合数据分布可以写成 KL 最小化。
5. **ELBO 与扩散模型：**理解路径分布的 KL、链式法则及训练目标。

进一步的严格证明见已有笔记：[Properties of KL Divergence](https://egnarochan.github.io/Egnaro/properties_of_kl_divergence.html)；返回[文章列表](https://egnarochan.github.io/Egnaro/index.html)。
