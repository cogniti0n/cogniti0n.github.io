---
title: Non-Convex Optimization in DL
date: 2025-08-08 05:50 +0800
description: Nonconvex optimization, loss landscape for overparameterized models.
categories: [optimization, ml-optimization]
tags: [online-learning]
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

In general, modern deep learning architectures are overparametrized and therefore non-convex. Optimizers such as SGD, in general, do not work well with nonconvex objective functions. In fact, _finding the global minimum of a general non-convex function is NP-hard_ in the sense that, we can construct global optimization problems that are provably NP-hard (see [2]). Moreover, it is NP-hard to check  Moreover, overparametrized deep learning models (unlike underparameterized models) have loss functions that are not convex even in an arbitrary local area near a minima. 


However, GS/SGD can find a good approximation of the global minima in most DL architectures (i.e., the training loss can be brought down to near zero). This suggest that deep learning problems are part of a subset that can be effectively solved using gradient-based optimizers, and also all local minima are approximately global minima. Here we state some results that build the foundations of such claims.

## Convergence to Local Minima

In general, for a function $f$, it is not true that $\nabla f = 0$ and $\nabla^2 f \succeq 0$ implies $x$ is a local minima. Therefore, we have to consider a stronger condition, known as the _strict-saddle_ condition[3]. 

> **Theorem** Suppose $f$ is a function that satisfies the following condition: $\exists\,\varepsilon_0,\tau_0,c > 0$ such that if $\vert f(x) \vert_2 \le \varepsilon < \varepsilon_0$ and $\nabla^2 f(x) \succeq \tau_0 I$, then $x$ is $\varepsilon^c$-close to a local minimum of $f$. Then GD (and SGD, etc.) can converge to a local minimum of $f$ up to $\delta$-error in Euclidean distance in time $poly(1/\delta, 1/\tau_0, d)$.

## When is GD Effective?

> **Definition** (Polyak-Łojasiewicz conditions) For a (possibly non-convex) function $f$, suppose a minimizer $x^* $ exists (need not be unique). Also suppose it satisfies the _$\mu$-PL_ inequality.
>
> $$
\frac{1}{2} \vert \nabla f(x) \vert^2_2 \ge \mu(f(x) - f(x^* ))
> $$
>
> Such a condition is called the Polyak-Łojasiewicz condition.

A useful lemma to give us insight:

> **Lemma** An $\alpha$-stronglyt convex function satisfies the $\alpha$-PL inequality.

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
f(x^k) - f(x^* ) \le \left(1-\frac{\mu}{\beta}\right)^k (f(x^0) - f(x^* ))
> $$

_proof_. From our choice of step size,

$$
f(x^{t+1}) \le f(x^t) - \frac{1}{2\beta} \vert \nabla f(x^t) \vert^2_2
$$

which yields the result. This is an important result, as PL conditions are weaker than strongly convex - moreover, it is not a 'global' property, like strong convexity. For strongly convex and smooth functions, gradient desccent converges linearly, which can also be applied to PL conditions.

Later I will look at [4] that extends this definition of PL conditions.


## References
[1] Ma, Tengyu. 2022. Lecture notes for CS229M (Machine Learning Theory) \
[2] Danilova, Marina, Pavel Dvurechensky, Alexander Gasnikov, Eduard Gorbunov, Sergey Guminov, Dmitry Kamzolov, and Innokentiy Shibaev. 2020. “Recent Theoretical Advances in Non-Convex Optimization.” arXiv [Math.OC]. arXiv. http://arxiv.org/abs/2012.06188. \
[3] Lee, Jason D., Max Simchowitz, Michael I. Jordan, and Benjamin Recht. 2016. “Gradient Descent Converges to Minimizers.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1602.04915. \
[4] Liu, Chaoyue, Libin Zhu, and Mikhail Belkin. 2022. “Loss Landscapes and Optimization in Over-Parameterized Non-Linear Systems and Neural Networks.” Applied and Computational Harmonic Analysis 59 (July): 85–116.

