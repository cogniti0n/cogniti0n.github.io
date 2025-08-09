---
title: Online Optimization Setup
date: 2025-08-06 05:50 +0800
description: Basic setup for online (convex) optimization and regret analysis.
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


The online optimization setup is a flexible framework to analyze iterative optimization methods. In this setup, at each time step $t$, the algorithm picks a point $x_t$ (the parameters of the model) where $x \in \mathcal{X} \subseteq \R^n$. Then a loss function $f_t(x)$ is given as a function of the parameters - therefore the loss is $f_t(x_t)$.

> **Definition** (Regret) The regret of the algorithm, after $T$ rounds, is defined as
>
> $$
R(T) = \sum^T_{t=1} f_t(x_t) - \min_{x \in \mathcal{X}} \sum^T_{t=1} f_t(x)
> $$

When devising an optimization algorithm, our aim is to ensure that 

$$
\limsup_{T \to \infty} R(T)/T \le 0
$$

The simplest algorithm is the standard greedy gradient descent algorithm:

> **Algorithm** (Greedy Projection) Select an arbitrary $x_0 \in \mathcal{X}$ and a sequence of learning rates $\eta_1, \eta_2, \dots \in \R^+$. At time step $t$, after receiving a convex cost function $f_t(x)$, select the next vector according to
> 
> $$
x_{t+1} = \Pi_\mathcal{X} (x_t - \eta_t \nabla f_t(x_t))
> $$
>
> Where $\Pi_\mathcal{X}$ is the projection operator 
$$\Pi_\mathcal{X}(y) =  \min_{x \in \mathcal{X}} || x-y ||$$
.

We first assume that the feasible set $F$ has bounded diameter (i.e., there exists some $D$ such that $\vert x-y \vert \le D$ for all  $x,y \in F$) and $\vert \nabla f_t(x) \vert_\infty$ is bounded for all $t$ and $x$. Then the following theorem holds.

> **Theorem** If the step size is chosen $\eta_t = 1/\sqrt{t}$, then the regret of greedy projection is bounded
>
> $$
R(T) \le \frac{ \vert F \vert^2 \sqrt{T}}{2} + \left( \sqrt{T} - \frac{1}{2}\right) \vert \nabla f \vert^2
> $$
>
> where $\vert \nabla f \vert = \sup_{x,t} \vert \nabla f_t(x) \vert$.

## References
[1] Zinkevich, Martin A. 2003. “Online Convex Programming and Generalized Infinitesimal Gradient Ascent.” International Conference on Machine Learning, August, 928–36. \
[2] Reddi, Sashank J., Satyen Kale, and Sanjiv Kumar. 2019. “On the Convergence of Adam and Beyond.” arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/1904.09237.
