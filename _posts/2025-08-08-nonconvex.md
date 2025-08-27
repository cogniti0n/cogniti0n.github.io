---
title: Deep Neural Networks and Overparameterization In Optimiziation
date: 2025-08-08 05:50 +0800
description: Some non-related topics on deep learning and optimization.
categories: [optimization, ml-optimization]
math: true
toc: false
---

$$
    \def\argmin{\mathop{\mathrm{argmin}}}
    \def\argmax{\mathop{\mathrm{argmax}}}
    \def\expectation{\mathop{\mathbb{E}}}
    \def\dom{\mathrm{dom}}
    \def\R{\mathbb{R}}
$$

In general, modern deep learning architectures are overparametrized and therefore non-convex. Optimizers such as SGD, in general, do not work well with nonconvex objective functions. In fact, _finding the global minimum of a general non-convex function is NP-hard_ in the sense that, we can construct global optimization problems that are provably NP-hard (see [2]). Moreover, it is NP-hard to check if a point is a local minima or not. Even worse, overparametrized deep learning models (unlike underparameterized models) have loss functions that are not convex even in an arbitrary local area near a minima.

Despite all of these problems arising in overparametrized models, GD/SGD can find a good approximation of the global minima in most DL architectures (i.e., the training loss can be brought down to near zero). This suggest that deep learning problems are part of a subset that can be effectively solved using gradient-based optimizers, and also all local minima are approximately global minima. Here we state some results that build the foundations of such claims.

## Non-Convexity of Loss Landscapes

Informally, an overparameterized neural network does not have an isolated global minima, i.e., there exists some other local minima in an arbitrary neighborhood around a local minima. To formulate this notion more clearly, consider a supervised learning task with dataset $$\mathcal{D} = \{ x_i,y_i \} ^n_{i=1}$$ where $x_i \in \R^d$ and $y_i \in \R$, and family of models $f(w;x)$. Our goal is to find $w^* $ such that

$$
f(w^*;x_i) \approx y_i, \quad i = 1,\dots,n
$$

We define a map $F : \R^m \to \R^n$ such that $F(w) = y = [y_1,\dots,y_n]^\top$. We omit dependence on $x_i$. Then our goal is to minimize a loss function $\mathcal{L}(w)$, which is constructed so that the solutions of $F(w) = y$ are the global minimizers of $\mathcal{L}$. We assume that the map $F$ is Lipschitz and smooth.

> **Proposition** For a overparametrized system $F : \R^m \to \R^n$ where $m > 2n$ (think of $m$ as the dimension of weights in a neural network, and $n$ as the number of data points) and a general loss function $\mathcal{L}$, let $w^* $ be a global minimizer of the loss function, i.e., $\mathcal{L}(w^* ) = 0$. If the following conditions meet, then $\mathcal{L}(w)$ is not convex in any neighborhood of $w^* $.
>
> - $\nabla \mathcal{L}(w^* ) \ne 0$
> - For at least one $i$, $\mathrm{rank}(\nabla^2 F_i(w^* )) > 2n$

For a local minima $w^* $, we generally do not expect the gradient to vanish. Therefore, the conditions of the proposition are well-established.

## When is GD Effective?

> **Definition** (Polyak-Łojasiewicz conditions) For a (possibly non-convex) function $f$, suppose a minimizer $x^* $ exists (need not be unique). Also suppose it satisfies the _$\mu$-PL_ inequality.
>
> $$
\frac{1}{2} \vert \nabla f(x) \vert^2_2 \ge \mu(f(x) - f(x^* ))
> $$
>
> Such a condition is called the Polyak-Łojasiewicz condition.

A useful lemma to give us insight:

> **Lemma** An $\alpha$-strongly convex function satisfies the $\alpha$-PL inequality.

_proof_. From stong convexity,

$$
f(y) \ge f(x) + \nabla f(x)^\top (y-x) + \frac{\alpha}{2} \vert x-y \vert^2_2
$$

Minimize with respect to $y$ on both sides. Then,

$$
f(x^* ) \ge f(x) - \frac{1}{2\alpha} \vert \nabla f(x) \vert^2_2
$$

which is precisely the $\alpha$-PL inequality. Now, we show that the PL conditions are sufficient to guarantee convergence of GD.

> **Theorem** Suppose $f$ is $\beta$-smooth and $\mu$-PL. Then GD with step size $\eta = 1/\beta$ converges linearly, i.e.,
>
> $$
f(x^k) - f(x^*) \le \left(1-\frac{\mu}{\beta}\right)^k (f(x^0) - f(x^*))
> $$

_proof_. From our choice of step size,

$$
f(x^{t+1}) \le f(x^t) - \frac{1}{2\beta} \vert \nabla f(x^t) \vert^2_2
$$

which yields the result. This is an important result, as PL conditions are weaker than strongly convex - moreover, it is not a 'global' property, like strong convexity. For strongly convex and smooth functions, gradient desccent converges linearly, which can also be applied to PL conditions.

## References

[1] Ma, Tengyu. 2022. Lecture notes for CS229M (Machine Learning Theory) \
[2] Danilova, Marina, Pavel Dvurechensky, Alexander Gasnikov, Eduard Gorbunov, Sergey Guminov, Dmitry Kamzolov, and Innokentiy Shibaev. 2020. “Recent Theoretical Advances in Non-Convex Optimization.” arXiv [Math.OC]. arXiv. http://arxiv.org/abs/2012.06188. \
[3] Lee, Jason D., Max Simchowitz, Michael I. Jordan, and Benjamin Recht. 2016. “Gradient Descent Converges to Minimizers.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1602.04915. \
[4] Liu, Chaoyue, Libin Zhu, and Mikhail Belkin. 2022. “Loss Landscapes and Optimization in Over-Parameterized Non-Linear Systems and Neural Networks.” Applied and Computational Harmonic Analysis 59 (July): 85–116.
