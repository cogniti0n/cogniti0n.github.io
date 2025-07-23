---
title: The Rademacher Complexity
date: 2025-07-17 20:00:00 +0800
description: Part 4 of notes on C229M. Bounding the excess risk using the Rademacher complexity.
categories: [machine-learning-deep-learning, ML Theory (C229M T. Ma)]
tags: [mldl]
math: true
toc: false
---

$$
    \def\argmin{\mathop{\mathrm{argmin}}}
    \def\argmax{\mathop{\mathrm{argmax}}}
    \def\expectation{\mathop{\mathbb{E}}}
    \def\R{\mathbb{R}}
$$

Before, we bounded the excess risk by

$$
\text{excess risk} \le \left|L(h^*)-\hat{L}(h^*)\right|+\sup_{h \in \mathcal{H}} \left[L(h) - \hat{L}(h)\right]
$$

For continuous hypothesis classes, we used a brute-force discretization method to bound the second term. However, in reality the bound we achieved is not enough. First, the bound we had - $O(\sqrt{p/n})$, requires that $n \gg p$ to achieve high performance (low excess risk). In reality, $n < p$ performs well quite often (with incredibly high dimensions). Also, the bound cannot distinguish between, for example, the following possible parameter spaces
$$
\{\theta:||\theta||_1 \le B\}
$$ 
and 
$$
\{\theta:||\theta||_2 \le B\}
$$
. Therefore, we need to find a better representation of a ‘complexity measure’ of $\Theta$, and find the relation between the complexity of the parameter space and the excess risk bound.

To solve the issues above, we introduce the *Rademacher complexity*. We first define the Rademacher random variable: $\sigma_i$ is sampled from $\{\pm 1\}$ with equal probability of $1/2$.

> **Definition** (Rademacher Complexity) Let $F$ be a family of real-valued functions that maps $Z \to \mathbb{R}$, where $Z$ is some 'data space'. Let $p$ be a distribution over $Z$ and let $z_i \in Z$ be i.i.d. random variables. Then, the (average) Rademacher complexity is defined as
>
>$$
R_n(F)\equiv \mathop{\mathbb{E}}_{z_i \sim i.i.d.\,p}\left[ \expectation_{\sigma_i\sim i.i.d.\,\{\pm1\}}\left[ \sup_{f \in F} \frac{1}{n}\sum^n_{i=1} \sigma_i f(z_i) \right] \right]
>$$

Qualitatively, we can understand the Rademacher complexity as the best possible correlation between $f(z_1),\dots,f(z_n)$ and $(\sigma_1,\dots,\sigma_n)$, i.e., ‘how well the losses of $n$ data can correlate with a random binary pattern’

>**Theorem** We achieve the following bound for i.i.d. random variables $z_i \sim p$.
>
>$$
\expectation_{z_i\sim i.i.d.\,p}\left[ \sup_{f \in F}\left[ \frac{1}{n} \sum^n_{i=1} f(z_i) - \mathbb{E}f(z) \right] \right] \le 2 R_n(F) \tag{1}
>$$

 If we understand $F$ as the ‘family of losses’, i.e., $f(z^{(i)})=l\left(\left(x^{(i)},y^{(i)}\right),h\right)$, then we can see that $\frac{1}{n}\sum^n_{i=1}f(z_i)=\hat{L}(h)$ and $\mathbb{E}f(z) = L(h)$. Therefore, we have established a connection with the excess risk.

Before we prove the theorem above, let’s establish an example, which will be used in the next post. Consider a binary classification task. Then, $y \in \{\pm 1\}$ and the loss is a $0-1$ loss, i.e.,

$$
l((x,y),h)=\mathbf{1}(h(x)\ne y)=\frac{1}{2}(1-y h(x))
$$

Then we can see that

$$
\begin{aligned}
R_n(F) &= \mathbb{E}\left[ \sup_{h \in \mathcal{H}} \frac{1}{n} \sum^n_{i=1} l\left(\left(x^{(i)},y^{(i)}\right),h\right)\sigma_i \right] \\ 
       &=\mathbb{E}\left[ \sup_{h \in \mathcal{H}} \frac{1}{n} \sum^n_{i=1} \frac{1}{2} \left( 1 - y^{(i)}h \left( x^{(i)} \right ) \right )\sigma_i \right] \\ 
       &= \mathbb{E}\left[ \sup_{h \in \mathcal{H}} \frac{1}{2n} \sum^n_{i=1} \left(-y^{(i)}h\left(x^{(i)}\right)\right)\sigma_i \right] \\ 
       &= \mathbb{E} \left[ \sup_{h \in \mathcal{H}} \frac{1}{2n} \sum_i h\left(x^{(i)}\right)\sigma_i' \right] = \frac{1}{2}R_n(\mathcal{H})
\end{aligned}
$$

Therefore, the Rademacher complexity of the family of loss functions are related to the complexity of the hypothesis class - which matches well with our intuition. Now we prove the theorem above. We first prove a simple lemma.

>**Lemma** Supremum and expectation value.
>
>$$
\sup_u \expectation_v \, g(u,v) \le \sup_u \mathbb{E}_v \left[ \sup_{\tilde{u}} g(\tilde{u},v) \right] = \mathbb{E}_v \left[ \sup_{\tilde{u}} g(\tilde{u},v) \right]
>$$

_proof_. The first inequality is trivial. The second equality holds because the term is independent of $u$.

Now for the actual proof.

_proof_. First we apply a symmetrization technique. Here we fix $z_1,\dots,z_n$ and resample $z_1’,\dots,z_n’$ from the same distribution.

$$
\begin{aligned}
\sup_{f \in F} \left( \frac{1}{n} \sum_i f(z_i) - \mathbb{E} f \right) &= \sup_{f \in F} \left[ \frac{1}{n} \sum_i f(z_i) - \expectation_{\{z_i'\}}\left[ \frac{1}{n} \sum^n_{i=1} f\left(z_i'\right) \right] \right] \\ 
&= \sup_{f \in F} \expectation_{\{z_i'\}} \left[ \frac{1}{n} \sum_i f(z_i) - \frac{1}{n} \sum_i f\left(z_i'\right) \right] \\
&\le \expectation_{\{z'\}} \sup_{f \in F} \left[ \frac{1}{n}\sum_i f(z_i) - \frac{1}{n}\sum_i f\left(z_i'\right) \right]
\end{aligned}
$$

Therefore, the following inequality holds.

$$
\expectation_{\{z_i\}}\left[\sup_{f \in F} \left[ \frac{1}{n}\sum f(z_i) - \mathbb{E}f \right]\right]  \le \expectation_{\{z_i, z_i'\}} \left[ \sup_{f \in F}\frac{1}{n} \sum^n_{i=1} \left(f(z_i) - f\left(z_i'\right)\right) \right]
$$

We use the fact that $f(z_i) - f\left(z_i'\right)$ has the same distribution as $f\left(z_i'\right)-f(z_i)$ and correspondingly the same distribution as $\sigma_i\left(f(z_i) - f\left(z_i'\right)\right)$, and thus we can find

$$
\expectation_{\{z_i, z_i'\}} \left[ \sup_{f \in F}\frac{1}{n} \sum^n_{i=1} \left(f(z_i) - f\left(z_i'\right)\right) \right] = \expectation_{\{z_i,z_i',\sigma_i\}} \left[ \sup_{f \in F} \frac{1}{n} \sum_i\sigma_i\left(f(z_i)-f\left(z_i'\right)\right) \right]
$$

Since $-\sigma_i$ has the same distribution as $\sigma_i$, we can bound the above equation to

$$
\expectation_{\{z_i,z_i',\sigma_i\}} \left[ \sup_{f \in F} \frac{1}{n} \sum_i\sigma_i\left(f(z_i)-f\left(z_i'\right)\right) \right] \le 2 \expectation_{\{z_i,\sigma_i\}}\left[ \sup_{f \in F} \frac{1}{n} \sum_i \sigma_i f(z_i) \right] = 2R_n(F)
$$

By bounding the maximum excess risk, we first removed $\mathbb{E}f$ (gained translational invariance) and also introduced randomness $\sigma_i$, which will in turn allow us to drop the randomness from $z_1,\dots,z_n$. The Rademacher complexity was defined as the expectation over variables $z_i$ (in our case, this corresponds to the data points). We now define the _empirical_ Rademacher complexity.

>**Definition** (Empirical Rademacher complexity) We drop the randomness with respect to $z_i$.
>
>$$
R_S(F)\equiv \expectation_{\{\sigma_i\}}\left[ \sup_{f \in F} \frac{1}{n} \sum_i \sigma_i f(z_i) \right]
>$$

where $$S = \{ z_1,\dots,z_n \}$$. Then, we see that $\expectation\left[R_S(F)\right] =R_n(F)$, and just like we bounded the excess risk by using concentration inequalities, we can bound the empirical Rademacher complexity. Before we do that, we first remove the expectation with respect to $z_i$ in Eq. (1).

>**Theorem** Suppose, $\forall f \in F$, $0 \le f(x) \le 1$. Then, with probability at least $1 -\delta$ over the randomness of $z_1,\dots,z_n$, 
>
>$$
\sup_{f \in F} \left[\frac{1}{n}\sum_i f(z_i) - \mathbb{E}f \right] \le 2 R_S(F) + 3\sqrt{\frac{\log(2/\delta)}{2n}}
>$$

*proof*. TODO

A useful aspect of the Rademacher complexity is that it is translation invariant.

>**Proposition** Let $F:Z\to\mathbb{R}$ be a family of functions. Define $F'=\{f(z)+c_0 : f \in F\}$ for a constant $c_0 \in \mathbb{R}$. Then $R_S(F) = R_S(F')$ and $R_N(F) = R_N(F')$.

_proof_. The proof is quite straightforward.

$$
\begin{aligned}
R_S(F') &= \expectation_{\{\sigma_i\}} \left[ \sup_{f' \in F'} \frac{1}{n} \sum^n_{i=1} \sigma_i f'(z_i) \right] \\
        &= \expectation_{\{\sigma_i\}} \left[ \sup_{f \in F} \frac{1}{n} \sum^n_{i=1} \sigma_i (f(z_i)+c_0) \right] \\
        &= \expectation_{\{\sigma_i\}} \left[ \sup_{f \in F} \frac{1}{n} \sum^n_{i=1} \sigma_i f(z_i) \right] = R_S(F) \\
\end{aligned}
$$

The last equation holds, as $$\expectation_{ \{ \sigma_i \} } \sigma_i = 0$$. Proving the proposition for the average Rademacher complexity is also straightforward. The proof is complete.

Now we bound the empirical Rademacher complexity using the dimension of $F$.

>**Theorem** Suppose $F$ satisfies the following. $\forall f \in F$, 
>$$
\frac{1}{n}\sum^n_{i=1}|f(z_i)|^2 \le M^2
>$$
>(which is a weaker version of $|f(z)| \le M$). Then, the following holds.
>
>$$
R_S(F)\le\sqrt{\frac{2M^2 \log |F|}{n}}
>$$

Which corresponds to the $\sqrt{p/n}$ in the discretization bound. 

_proof_. TODO

## References
[1] Ma, Tengyu. 2022. Lecture notes for CS229M (Machine Learning Theory)