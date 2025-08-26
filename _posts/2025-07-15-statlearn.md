---
title: Statistical Learning Theory
date: 2025-07-15 23:00:00 +0800
description: Preliminaries to Supervised Learning Theory - Statistical Learning Theory
categories: [machine-learning-deep-learning, classic-ml-theory]
math: true
toc: false
---

$$
    \def\argmin{\mathop{\mathrm{argmin}}}
    \def\argmax{\mathop{\mathrm{argmax}}}
    \def\expectation{\mathop{\mathbb{E}}}
    \def\R{\mathbb{R}}
$$

## Bayes Risk and Bayes Predictor

Before, we defined the excess risk as

$$
E(h)\equiv L(h) - \inf_{g \in \mathcal{H}} L(g)
$$

We did not mention it, but when exactly does the expected risk reach minimum? First use the law of total expectation to achieve

$$
L(h) = \expectation \left[ l(y,h(x)) \right] = \expectation \left[ \expectation \left[ l(y,h(x)) | x \right] \right]
$$

which we can rewrite as

$$
L(h) = \expectation_{x' \sim p} \left[ \expectation\left[ l(y,h(x')) | x=x' \right] \right]
$$

If we define tha _conditional risk_
$$r(z|x') = \expectation \left[ l(y,z) | x=x' \right]$$
, then the risk can be expressed as the sum (or integral) dependent on a single value of $f$. Therefore, minimizing the expected risk can be thought of as finding the _pointwise_ minimum $h(x')$ for any $x' \in X$ independently - furthermore, such a function value $h(x')$ must be chosen as the minimizer $z$ of $r(z|x')$.

> **Proposition** (Bayes predictor and Bayes risk) The expected risk is minimum at a Bayes predictor $ h_* : X \to Y $, satisfying for all $x' \in X$,
>
> $$
f_* (x') \in \argmin_{z \in Y} \, \expectation \left[ l(y,z) | x=x' \right] = \argmin_{z \in Y} \, r(z|x')
> $$
>
> The Bayes risk $ L^* $ is the risk of all Bayes predictors and is equal to
>
> $$
L^* = \expectation_{x' \sim p} \left[ \inf_{z \in Y} \left[ l(y,z) | x=x' \right] \right]
> $$

_proof_. See that

$$
L(h) - L^* = L(h) - L( h_* ) = \int_X \left[ r(f(x')|x) - \min_{z \in Y} r(z|x') \right] \, dp(x')
$$

which shows the proposition.

Some common Bayes predictors:

- Binary classification with $0-1$ loss

$$
f_* (x') \in \argmin_{z \in \{-1,1\} } \, \mathrm{Pr} \left[ y \ne z | x \ne x' \right] =  \argmax_{z \in \{-1,1\} } \, \mathrm{Pr}\left[ y=z | x=x' \right]
$$

- Regression with $l(y,z) = (y-z)^2$

$$
f_* (x') \in \argmin_{z \in \R} \left\{  \expectation\left[ (y- \expectation\left[y|x=x'\right]) \right] + (z - \expectation\left[y|x=x'\right])\right\}
$$

## PAC Learning

Our goal, in traditional statistical learning theory, is to bound the excess risk. The PAC learning framework lets us formulate a rigorous way of bounding errors.

> **Definition** (PAC-learning) A hypothesis class $\mathcal{H}$ is said to be PAC-learnable if there exists an algorithm $\mathcal{A}$ and a polynomial function $poly(\cdot,\cdot,\cdot,\cdot)$ such that for any $\varepsilon > 0$ and $\delta > 0$, for all distributions $\mathcal{D}$ on $X$ and for any target hypothesis $h \in \mathcal{H}$, the following holds for any sample size $m \ge poly(1/\varepsilon,1/\delta,n,size(h))$
>
> $$
\mathop{\mathrm{Pr}}_{\theta \sim \mathcal{D}^m} \, [ L(h_\theta) \le \varepsilon ] \ge 1-\delta
> $$
>
> If $\mathcal{A}$ runs in $poly(1/\varepsilon,1/\delta,n,size(c))$, then $\mathcal{H}$ is said to be efficiently PAC-learnable. When such an algorithm $\mathcal{A}$ exists, it is called a PAC-learning algorithm.

In a PAC-learning situation, the training and test examples are drawn from the same distribution $\mathcal{D}$, but we do not make further assumptions about $\mathcal{D}$.

Then naturally there are two ways of viewing statistical learning algorithms. One is PAC-learning mentioned above, and another is bounding the expectation of the risk. In most cases, one bound can be converted to another in a concise manner - use concentration inequalities, e.g., the Hoeffding inequality or the McDiarmid inequality.

For instance, suppose we have some bound on the expectation value

$$
\expectation\left[ \sup_{h \in \mathcal{H}}\left\{ L(h) - \hat{L}(h) \right\} \right]
$$

Then we can use McDiarmid to achieve a PAC-bound. For data points $z_i = (x_i, y_i)$, let

$$
H(z_1,\dots,z_N) = \sup_{h \in \mathcal{H}} \left\{ L(h) - \hat{L}(h)\right\}
$$

Now we add the assumption that $0 \le l(h(x),y) \le l_\infty$ for all $h$ and $(x,y)$.

We first show that the bounded difference condition holds, i.e.,

$$
|H(z_1, \dots, z_{i-1},z_i,z_{i+1}, \dots, z_N) - H(z_1, \dots, z_{i-1}, z_i', z_{i+1}, \dots, z_N)| \le c_i
$$

First denote $\mathcal{D} = (z_1, \dots, z_i, \dots, z_N)$ and $\mathcal{D}' = (z_1, \dots, z_i', \dots, z_N)$. Now note that, (with slight abuse of notation)

$$
\hat{L}(h,\mathcal{D}') - \hat{L}(h,\mathcal{D}) = \frac{1}{N} \left( l(h(x_i'), y_i) - l(h(x_i), y_i) \right) \le \frac{l_\infty}{N}
$$

Therefore,

$$
\begin{aligned}
H(\mathcal{D}) - H(\mathcal{D}') &= \sup_{h \in \mathcal{H}} \left\{ L(h) - \hat{L}(h,\mathcal{D}') + \hat{L}(h,\mathcal{D}') - \hat{L}(h,\mathcal{D}) \right\} - \sup_{h \in \mathcal{H}} \left\{ L(h) - \hat{L}(h,\mathcal{D}') \right\} \\
&\le \sup_{h \in \mathcal{H}} \left\{ L(h) - \hat{L}(h,\mathcal{D}') \right\} + \sup_{h \in \mathcal{H}} \left\{ \hat{L}(h,\mathcal{D}') - \hat{L}(h,\mathcal{D}) \right\} - \sup_{h \in \mathcal{H}} \left\{ L(h) - \hat{L}(h,\mathcal{D'}) \right\} \\
&= \sup_{h \in \mathcal{H}} \left\{ \hat{L}(h,\mathcal{D}') - \hat{L}(h,\mathcal{D}) \right\} \le \frac{l_\infty}{N}
\end{aligned}
$$

From a symmetric argument, we can bound the difference
$$|H(\mathcal{D}) - H(\mathcal{D}')| \le l_\infty / N$$
. Finally from McDiarmid, we have the following bounds.

$$
\sup_{h \in \mathcal{H}}\left\{ L(h) \le \hat{L}(h) \right\} \le \expectation\left[ \sup_{h \in \mathcal{H}} \left\{ L(h) - \hat{L}(h) \right\} \right] + l_\infty \sqrt{\frac{\log(1/\delta)}{2N}}
$$

$$
\sup_{h \in \mathcal{H}}\left\{ \hat{L}(h) - L(h) \right\} \le \expectation\left[ \sup_{h \in \mathcal{H}} \left\{ \hat{L}(h) - L(h) \right\} \right] + l_\infty \sqrt{\frac{\log(1/\delta)}{2N}}
$$

with probability $1 - \delta$ (a PAC bound!!). Finally, we have the union bound

$$
\begin{aligned}
\sup_{h \in \mathcal{H}}&\left\{ \hat{L}(h) - L(h) \right\} + \sup_{h \in \mathcal{H}}\left\{ L(h) - \hat{L}(h) \right\} \\ \le \, &\expectation\left[ \sup_{h \in \mathcal{H}} \left\{ L(h) - \hat{L}(h) \right\} \right] + \expectation\left[ \sup_{h \in \mathcal{H}} \left\{ \hat{L}(h) - L(h) \right\} \right] + l_\infty \sqrt{\frac{2 \log (2/\delta)}{N}}
\end{aligned}
$$

with probability $1 - \delta$.

## References

[1] Bach, F. (2024). Learning theory from first principles. MIT Press. \
[2] Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018). Foundations of machine learning (2nd ed.). MIT Press.
