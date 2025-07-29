---
title: Statistical Learning Theory
date: 2025-07-15 23:00:00 +0800
description: Preliminaries to Supervised Learning Theory - Statistical Learning Theory
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

## Statistical Learning Theory

We generally define the notion of statistical learning theory. Given data $\mathcal{D} = \{ (x_1,y_1),\dots,(x_n,y_n) \}$, obtained i.i.d. from some random distribution $p$, we want to find a mapping $\mathcal{A}$ from $\mathcal{D}$ to a function (or hypothesis) from $X$ to $Y$ that minimized the excess expected risk

$$
L\left(\mathcal{A}(\mathcal{D})\right) - L^*
$$

when $p$ is unknown. Since $\mathcal{D}$ is random, we need to deal with such randomness. One way is to take the expectation with respect to the randomness and see if the excess risk goes to zero as $n \to \infty$, i.e.,

$$
\lim_{n \to \infty} \left[ \expectation \left[ L\left(\mathcal{A}(\mathcal{D})\right) \right] - L^* \right] = 0
$$

Another way to view this problem is _probably approximately correct (PAC) learning_. For a given $\delta \in (0,1)$ and $\varepsilon > 0$,

$$
\mathrm{Pr}\left[ L\left(\mathcal{A}(\mathcal{D})\right) - L^* \le \varepsilon \right] \le 1 - \delta
$$

The goal of learning theory in this sense is to find an $\varepsilon$ that is as small as possible. 

An algorithm is called _universally consistent_ if for all all probability distributions $p$ on $(x,y)$, $\mathcal{A}$ is consistent in expectation for the distribution $p$. Most often, we want to study uniform consistency within a class $\mathcal{P}$ of distributions satisfying some regularity property. Therefore, we aim at finding an algorithm

$$
\mathcal{A} = \argmin \, \sup_{p \in \mathcal{P}} \left\{ \expectation \left[ L\left(\mathcal{A}(\mathcal{D})\right) - L^* \right] \right\}
$$

and the corresponding risk is called the _minimax risk_. We want to bound the excess risk so esimate the minimax risk.

## References
[1] Bach, Francis. 2024. Learning Theory from First Principles. MIT Press.