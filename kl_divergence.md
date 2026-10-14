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

## 2. KL 散度的直观意义：平均额外对数损失

考虑一个概率预测问题：随机变量 $X$ 的真实分布为 $P$，我们却使用模型分布 $Q$ 预测它。对观测结果 $x$，模型给出的概率为 $Q(x)$。

如何评价这次预测？一种标准选择是**对数损失（log loss）**：

$$
\ell_Q(x)=-\log Q(x).
$$

预测越自信且越正确，损失越小；若模型把实际发生的结果判为几乎不可能，损失就非常大。使用对数还有一个重要性质：若两个独立事件的联合概率是概率之积，联合对数损失便等于两个损失之和。

由于观测 $x$ 由真实分布 $P$ 产生，评价模型的**平均对数损失**必须对 $P$ 取期望：

$$
L_P(Q)=\mathbb{E}_{x\sim P}[-\log Q(x)].
$$

如果我们恰好知道真实分布并用它来预测，平均损失是 $L_P(P)$。两者之差为

$$
\begin{aligned}
L_P(Q)-L_P(P)
&=\mathbb{E}_{x\sim P}
\left[-\log Q(x)+\log P(x)\right]\\
&=\mathbb{E}_{x\sim P}
\left[\log\frac{P(x)}{Q(x)}\right].
\end{aligned}
$$

这恰好就是第 1 节定义的 KL 散度。因此，

因此，可以把 $D_{\mathrm{KL}}(P\|Q)$ 理解为：**数据来自 $P$，但使用 $Q$ 预测时，相比使用真实分布 $P$，平均多承担多少对数损失。**

这里的顺序非常重要：**$P$ 决定数据出现的频率，$Q$ 决定预测给这些数据分配的概率**。KL 不是简单地逐点比较两组数字，再求一个普通的距离。

## 3. 例子：两种硬币预测分布

假设硬币实际上以 $0.8$ 的概率出现正面，以 $0.2$ 的概率出现反面，但模型认为正、反面的概率分别是 $0.6$ 和 $0.4$：

$$
P=(0.8,0.2),\qquad Q=(0.6,0.4).
$$

对于**正面**，模型低估了概率，其额外对数损失是

$$
\log\frac{0.8}{0.6}=\log\frac43\approx0.2877.
$$

对于**反面**，模型高估了概率，此时额外对数损失反而为负：

$$
\log\frac{0.2}{0.4}=\log\frac12\approx-0.6931.
$$

但是正面会出现 $80\%$ 的时间，反面只会出现 $20\%$ 的时间。于是平均额外损失为

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=0.8\log\frac{0.8}{0.6}
+0.2\log\frac{0.2}{0.4}\\
&\approx0.0915.
\end{aligned}
$$

**需要区分单次预测与平均预测：**某次事件上，错误模型的对数损失可能比真实分布更小；但是从真实分布反复抽样，平均额外损失不可能为负。后面将严格证明这一点。

如果交换 $P$ 和 $Q$，得到

$$
D_{\mathrm{KL}}(Q\|P)
=0.6\log\frac{0.6}{0.8}
+0.4\log\frac{0.4}{0.2}
\approx0.1046.
$$

数值不同，是因为交换以后，**用来加权各个事件的概率也变了**。这就是 KL 散度不对称的根本原因。

## 4. KL 的非负性：为什么平均损失不会更小？

KL 散度满足 **Gibbs 不等式**：

$$
D_{\mathrm{KL}}(P\|Q)\geq0,
$$

且等号成立当且仅当 $P=Q$（连续情形为几乎处处相等）。这是我们称它为“平均额外损失”的数学依据。

**证明。** 若存在 $P(x)>0,Q(x)=0$，KL 为无穷大，结论显然。否则，利用 $-\log u$ 的凸性及 Jensen 不等式：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\mathbb{E}_{x\sim P}
\left[-\log\frac{Q(x)}{P(x)}\right]\\
&\geq-\log\mathbb{E}_{x\sim P}
\left[\frac{Q(x)}{P(x)}\right]\\
&=-\log\sum_{x:P(x)>0}Q(x)\\
&\geq0.
\end{aligned}
$$

最后一步利用了 $Q$ 的总概率为 $1$。要使两个不等式同时取等，$Q$ 必须把全部概率放在 $P$ 的支撑集上，而且 $Q(x)/P(x)$ 在该集合上为常数；归一化条件进一步给出 $Q=P$。

**不要从每项非负去理解这个证明。** 单项 $\log(P(x)/Q(x))$ 可以为负；非负性是整个期望和概率归一化共同作用的结果。

KL 不是数学意义上的距离：一般而言 $D_{\mathrm{KL}}(P\|Q)\neq D_{\mathrm{KL}}(Q\|P)$，也不满足三角不等式。

## 5. 熵、交叉熵以及最大似然估计

### 5.1 熵与交叉熵如何联系 KL？

如果用真实分布 $P$ 自己预测数据，平均对数损失称为**熵（entropy）**：

$$
H(P)=-\mathbb{E}_{x\sim P}\log P(x).
$$

如果使用另一个分布 $Q$ 来预测，同样在真实数据 $P$ 上计算平均损失，得到**交叉熵（cross-entropy）**：

$$
H(P,Q)=-\mathbb{E}_{x\sim P}\log Q(x).
$$

因此，按照第 2 节的推导，立刻得到

$$
D_{\mathrm{KL}}(P\|Q)=H(P,Q)-H(P).
$$

这三个概念的区别非常明确：**熵**衡量 $P$ 本身的平均不确定性；**交叉熵**衡量用 $Q$ 预测来自 $P$ 的数据时的平均损失；**KL**则是后者相对于前者增加的损失。

### 5.2 为什么最小化交叉熵等于最小化 KL？

令 $P=p_{\mathrm{data}}$ 是真实数据分布，$Q=p_\theta$ 是参数为 $\theta$ 的模型分布。因为 $H(p_{\mathrm{data}})$ 与 $\theta$ 无关，所以

$$
\begin{aligned}
\arg\min_\theta
D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)
&=\arg\min_\theta H(p_{\mathrm{data}},p_\theta)\\
&=\arg\max_\theta
\mathbb{E}_{x\sim p_{\mathrm{data}}}
\log p_\theta(x).
\end{aligned}
$$

真实分布未知，因此我们用训练样本 $x^{(1)},\ldots,x^{(N)}$ 的经验平均来估计最后一个期望：

$$
\mathcal L_{\mathrm{NLL}}(\theta)
=-\frac1N\sum_{i=1}^{N}\log p_\theta(x^{(i)}).
$$

最小化负对数似然（negative log-likelihood，NLL），就是**最大似然估计（maximum likelihood estimation，MLE）**。因此，MLE 的总体目标可以视为最小化 $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)$；训练集上的 NLL 则是相应期望的经验估计。这里不需要把连续数据的经验分布直接代入 KL 公式。

这一结果还解释了监督分类中的交叉熵：如果标签分布是 one-hot，那么给正确类别赋予概率 $p_\theta(y\mid x)$ 的交叉熵就是 $-\log p_\theta(y\mid x)$。

## 6. Forward KL 与 Reverse KL

固定目标分布 $P$，让模型分布 $Q_\theta$ 随参数变化。有两个不同的优化问题：

$$
\underbrace{D_{\mathrm{KL}}(P\|Q_\theta)}_{\text{Forward KL}}
\qquad\text{和}\qquad
\underbrace{D_{\mathrm{KL}}(Q_\theta\|P)}_{\text{Reverse KL}}.
$$

在 Forward KL 中，我们对 $x\sim P$ 取期望，所以所有真实分布经常出现的区域都会影响目标；如果 $Q_\theta(x)=0$ 而 $P(x)>0$，损失无穷大。Reverse KL 则对 $x\sim Q_\theta$ 取期望；如果模型把质量放在 $P(x)=0$ 的区域，损失无穷大。

**常见的多模态直觉。** 若 $P$ 有多个相距很远的模态，但模型族受到限制（例如只能用单峰分布近似），最小化 Forward KL 往往倾向覆盖多个模态（mode-covering），而最小化 Reverse KL 往往倾向集中到某一个模态（mode-seeking）。理由并非“KL 本身具有某种模式偏好”，而是**两种方向的惩罚和取期望区域不同，加上模型族无法精确表达 $P$**。

这个判断有两个边界：一是并非对任意分布或模型族都成立；二是如果模型族包含 $P$，则两者共同的全局最优都是 $Q_\theta=P$，KL 值为 $0$。实际论文中描述 mode-seeking 时，应当明确模型表达能力与支撑集的假设。

## 7. KL 的链式法则与数据处理不等式

### 7.1 联合分布为什么可以分解？

设 $P(X,Y)$ 和 $Q(X,Y)$ 是同一对随机变量的两个联合分布。由条件概率分解，

$$
P(x,y)=P(x)P(y\mid x),
\qquad
Q(x,y)=Q(x)Q(y\mid x).
$$

把这两个式子代入对数比中：

$$
\begin{aligned}
\log\frac{P(x,y)}{Q(x,y)}
&=\log\frac{P(x)}{Q(x)}
+\log\frac{P(y\mid x)}{Q(y\mid x)}.
\end{aligned}
$$

在 $P(x,y)$ 下对等式取期望，就得到**KL 的链式法则**：

$$
\begin{aligned}
D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))
&=D_{\mathrm{KL}}(P(X)\|Q(X))\\
&\quad+\mathbb{E}_{x\sim P(X)}
\left[
D_{\mathrm{KL}}(P(Y\mid x)\|Q(Y\mid x))
\right].
\end{aligned}
$$

它说的不是“联合 KL 大约等于逐步 KL”，而是一个**精确分解**。其中条件 KL 的外层期望必须由第一个分布 $P(X)$ 来计算。

### 7.2 边缘化与数据处理

条件 KL 非负，因此

$$
D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))
\geq D_{\mathrm{KL}}(P(X)\|Q(X)).
$$

这意味着：对两个分布都丢弃变量 $Y$，不会使 KL 增大。更一般地，如果两个输入分布 $P_X,Q_X$ 通过**相同**的随机映射 $K(y\mid x)$ 得到 $P_Y,Q_Y$，则

$$
D_{\mathrm{KL}}(P_Y\|Q_Y)
\leq D_{\mathrm{KL}}(P_X\|Q_X).
$$

这个结论叫作 **数据处理不等式（data processing inequality）**：同一数据处理操作不会使两个分布在 KL 意义下变得更加可区分。

### 7.3 多步生成轨迹的链式法则

对一条轨迹 $x_{0:T}$，假设 $P,Q$ 按时间顺序分解：

$$
P(x_{0:T})=P(x_0)\prod_{t=1}^T P(x_t\mid x_{<t}),
$$

$$
Q(x_{0:T})=Q(x_0)\prod_{t=1}^T Q(x_t\mid x_{<t}).
$$

其中 $x_{<t}=(x_0,\ldots,x_{t-1})$。反复使用链式法则：

$$
\begin{aligned}
&D_{\mathrm{KL}}(P(x_{0:T})\|Q(x_{0:T}))\\
&=D_{\mathrm{KL}}(P(x_0)\|Q(x_0))\\
&\quad+\sum_{t=1}^T
\mathbb E_{x_{<t}\sim P}
\left[
D_{\mathrm{KL}}\left(
P(x_t\mid x_{<t})\|Q(x_t\mid x_{<t})
\right)
\right].
\end{aligned}
$$

**研究中容易忽略的细节：**这里的每个条件 KL 都是对 $P$ 的前缀分布取期望。若把它随意替换为模型自己生成的前缀分布，就不再是上述 Forward KL 的严格分解。

## 8. 从 KL 推导 ELBO

### 8.1 为什么需要引入辅助分布？

潜变量生成模型定义联合分布 $p_\theta(x,z)$，但我们通常只观测到 $x$，希望最大化其边缘似然：

$$
p_\theta(x)=\int p_\theta(x,z)\,dz.
$$

这个积分可能无法精确计算，因此引入一个可计算的辅助分布 $q_\phi(z\mid x)$，尝试得到 $\log p_\theta(x)$ 的下界。为使下面的 KL 和对数期望有意义，假设 $p_\theta(x)>0$，且 $q_\phi(z\mid x)$ 对真实后验 $p_\theta(z\mid x)$ 绝对连续。

### 8.2 精确恒等式先于下界

根据 Bayes 公式：

$$
\log p_\theta(z\mid x)
=\log p_\theta(x,z)-\log p_\theta(x).
$$

将它代入两个后验分布之间的 KL：

$$
\begin{aligned}
&D_{\mathrm{KL}}\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right)\\
&=\mathbb E_{q_\phi}
\left[
\log q_\phi(z\mid x)-\log p_\theta(z\mid x)
\right]\\
&=\log p_\theta(x)
-\mathbb E_{q_\phi}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
\end{aligned}
$$

定义

$$
\mathcal E(\theta,\phi;x)
:=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
$$

便得到精确等式：

$$
\log p_\theta(x)
=
\mathcal E(\theta,\phi;x)
+
D_{\mathrm{KL}}\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right).
$$

因为 KL 非负，所以 $\mathcal E(\theta,\phi;x)\leq\log p_\theta(x)$，即 $\mathcal E$ 是 **evidence lower bound（ELBO）**。这一推导同时揭示了下界的**精确松弛量**：它就是辅助后验与真实后验之间的 KL，而不是一个模糊的近似误差。

### 8.3 “重建项 − KL”从哪里来？

若联合模型分解为 $p_\theta(x,z)=p(z)p_\theta(x\mid z)$，则

$$
\begin{aligned}
\mathcal E(\theta,\phi;x)
&=\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]\\
&\quad-
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\|p(z)\right).
\end{aligned}
$$

这是 VAE 中常见的表达式。请不要混淆这里的两个 KL：与**真实后验**的 KL 衡量下界松弛；与**先验**的 KL 是对 ELBO 的一种代数分解。

也可以用 Jensen 不等式直接得到 ELBO；但从上述精确恒等式推导，更容易看清**下界究竟松在哪里**。

## 9. 扩散模型为何出现逐时间步 KL？

### 9.1 固定的前向过程与学习的反向过程

在标准离散时间扩散模型中，$x_0$ 是数据，前向 Markov 链逐步把它变成 $x_T$：

$$
q(x_{1:T}\mid x_0)
=\prod_{t=1}^T q(x_t\mid x_{t-1}).
$$

模型则从先验 $p(x_T)$ 开始，沿反向时间生成：

$$
p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t).
$$

这里 $q$ 是预先定义的加噪机制，$p_\theta$ 才是待学习的反向模型。

### 9.2 从 ELBO 得到变分训练损失

固定观测 $x_0$，将前向轨迹 $q(x_{1:T}\mid x_0)$ 作为辅助分布。第 8 节的 ELBO 立即给出：

$$
\begin{aligned}
-\log p_\theta(x_0)
&\leq
\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\log\frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}
\right]\\
&=:\mathcal L_{\mathrm{VLB}}(x_0).
\end{aligned}
$$

这里 $\mathcal L_{\mathrm{VLB}}$ 是**负对数似然的上界**；等价地，$-\mathcal L_{\mathrm{VLB}}$ 是对数似然的下界。

前向 Markov 链可以重新分解为

$$
q(x_{1:T}\mid x_0)
=q(x_T\mid x_0)
\prod_{t=2}^T q(x_{t-1}\mid x_t,x_0).
$$

其中出现的“反向后验”可以从已知前向过程计算：

$$
q(x_{t-1}\mid x_t,x_0)
=
\frac{
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)
}{
q(x_t\mid x_0)
}.
$$

把两个轨迹分解代入上界，并对对数乘积逐项整理，得到

$$
\begin{aligned}
\mathcal L_{\mathrm{VLB}}(x_0)
&=
D_{\mathrm{KL}}\left(q(x_T\mid x_0)\|p(x_T)\right)\\
&\quad+\sum_{t=2}^T
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

这就是**先验匹配项 + 逐时间步反向 KL 项 + 重建项**的来源。如果先验和前向过程固定，第一项本身不依赖待训练的反向参数 $\theta$。

核心逻辑是：先对**整条轨迹**构造 ELBO，然后利用 Markov 结构和 KL 链式分解获得逐步目标。逐步 KL 并不是凭直觉独立拼接出来的。

### 9.3 对 Masked Diffusion 的启示

Masked diffusion 同样从明确的前向遮罩过程出发，因此能够讨论相应的后验和变分目标。在特定转移结构、参数化与时间权重下，相关损失可进一步化为加权交叉熵。

但**一个 loss 写成交叉熵，不代表它自动就是某个 ELBO**。要建立这种结论，必须给出联合生成模型、辅助路径分布，并证明损失与其变分目标之间的关系。

## 10. 序列生成与可编辑轨迹：两个不能混淆的目标

### 10.1 自回归训练中的条件 KL

如果真实序列分布为 $P(x_{1:T})$，模型为 $Q_\theta(x_{1:T})$，那么链式法则给出

$$
\begin{aligned}
&D_{\mathrm{KL}}(P(x_{1:T})\|Q_\theta(x_{1:T}))\\
&=\sum_{t=1}^T
\mathbb E_{x_{<t}\sim P}
\left[
D_{\mathrm{KL}}\left(
P(x_t\mid x_{<t})
\|Q_\theta(x_t\mid x_{<t})
\right)
\right].
\end{aligned}
$$

所以，基于真实前缀的 teacher-forcing 最大似然训练确实对应总体 Forward KL 的一个精确分解（忽略有限样本估计误差）。但这并不意味着训练时的真实前缀分布与推理时的模型前缀分布相同。

### 10.2 最终输出分布不等于生成轨迹分布

对能够修改已生成位置的模型，设完整轨迹为 $\tau$，最终输出为 $x=f(\tau)$。模型给出轨迹分布 $p_\theta(\tau)$，于是输出分布为

$$
p_\theta(x)
=
\sum_{\tau:\,f(\tau)=x}p_\theta(\tau).
$$

对**最终输出**建模，目标可能是 $\max_\theta\log p_\theta(x)$；对**完整轨迹**建模，目标可能是匹配某个参考轨迹分布。这是两个不同问题。

因为最终输出是轨迹的一个映射，数据处理不等式表明

$$
D_{\mathrm{KL}}(P(\tau)\|Q(\tau))
\geq D_{\mathrm{KL}}(P(x)\|Q(x)).
$$

因此，轨迹分布匹配是一种通常更强的要求；最终分布匹配并不自动保证中间修改过程相同。

### 10.3 没有前向加噪，也能形式上写出 ELBO 吗？

**可以，但写出下界不等于得到了有用的训练目标。** 只要模型定义了完整轨迹概率 $p_\theta(\tau)$，并引入一个只支持 $f(\tau)=x$ 的辅助分布 $r_\phi(\tau\mid x)$，就可以得到

$$
\begin{aligned}
\log p_\theta(x)
&\geq
\mathbb E_{\tau\sim r_\phi(\tau\mid x)}
\left[
\log p_\theta(\tau)-\log r_\phi(\tau\mid x)
\right].
\end{aligned}
$$

相应的精确间隙为

$$
D_{\mathrm{KL}}\left(
r_\phi(\tau\mid x)\|
p_\theta(\tau\mid x)
\right).
$$

这里的 $r_\phi$ **不需要来自预先设定的前向加噪过程**。真正困难的是：如何定义或近似这个辅助路径分布，使下界既可计算、又足够紧，同时让训练采样的状态与模型实际生成时遇到的状态一致。

这也说明，在讨论可编辑生成过程的新 loss 时，应该先回答：**目标是最终输出的对数似然、一个有明确松弛量的变分下界，还是人为选择的逐步替代损失？** 三者不能仅凭形式相似就视为等价。

## 11. 核心结论与阅读路线

从上面的推导可以抽取出四条关系：

**对数损失：** KL 是用 $Q$ 预测真实分布 $P$ 时，相对于用 $P$ 预测的平均额外对数损失。

$$
D_{\mathrm{KL}}(P\|Q)=H(P,Q)-H(P).
$$

**最大似然：** 在真实数据分布固定时，最大化期望对数似然等价于最小化 Forward KL。

$$
\max_\theta\mathbb E_{p_{\mathrm{data}}}\log p_\theta(x)
\;\Longleftrightarrow\;
\min_\theta D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta).
$$

**链式法则：** 联合分布的 KL 可以精确分解为边缘 KL 和条件 KL，但条件 KL 的期望必须使用第一个分布。

**变分推断：** ELBO 与对数似然的差距是一个明确定义的后验 KL：

$$
\log p_\theta(x)-\mathrm{ELBO}
=
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right).
$$

建议理解顺序：先掌握第 1–5 节中的概率预测与对数似然，再看第 6–7 节中的优化方向和链式法则，最后用第 8–10 节理解变分目标、扩散训练与可编辑生成过程。

更形式化的 KL 性质及原有证明见 [Properties of KL Divergence](https://egnarochan.github.io/Egnaro/properties_of_kl_divergence.html)。返回 [文章列表](https://egnarochan.github.io/Egnaro/index.html)。
