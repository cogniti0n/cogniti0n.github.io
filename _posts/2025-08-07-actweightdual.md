---
title: Flatter Minima & Smaller Weight Norm Effects on Generalization
date: 2025-08-07 05:50 +0800
description: Summary of Feng, Yu, et al. 2023. Nat. Mach. Intell. 5 (8) 908–18.
categories: [machine-learning-deep-learning, deep-learning]
tags: [paper-reading]
math: true
toc: false
---

## Some Minima are Better Than Others

In the overparameterized regime of deep learning, there are many local minima that fit the training data equally well. Out of those local minima, the consensus is that 1) flat minima (measured roughly using the eigenvalues of the Hessian) and 2) small-weight norm solutions generalize well. However, the problem with measuring the flatness of minima is that it differs for different parameterizations. Indeed, given two parameters related by $\theta = g(\eta)$, the Hessian for the loss function $L$ is related by

$$
(\nabla^2 L_\eta) (\eta) = (\nabla g)(\eta)^\top (\nabla^2 L)\left(g(\eta)\right)(\nabla g)(\eta)
$$

And therefore the flatness depends on what parameterization we use - this suggests that a more robust way of categorizing 'better minima' is required.

Here the authors find connections between the generalization gap and flatness/weight norm using a techique called _activity-weight duality_. 

## Activity-Weight Duality

Denote an $S$-layer fully connected neural network's weights as $W = (w^1, \dots, w^S)$, and assume that the training and test datasets are fixed, then the generalization gap is defined as

$$
\Delta l(W) = \langle l_k \rangle_{k \in D_\text{te}} - \langle l_k \rangle_{k \in D_\text{tr}}
$$

The main idea is, given $W$ and training data $x$, test data $x'$, is to find a dual weight $W'$ that satisfies $l(x',W) = l(x,W')$. Then we can analyze the loss landscape dependence on the weight $W$ while keeping data $x$ fixed. 

For a single layer $s-1$ and $s$, our goal is to find $w'$ such that

$$
\sum_i {w'}_{ij}^{s} a^{s-1}_i = \sum_i w_{ij}^s {a'}_i^{s-1}
$$

where $a_i^s$ is the activation of the $i$-th neuron of layer $s$. Because of overparametrization, there are infinitely many solutions for $w'$. Among these solutions, we want to find a solution that is closest to $w$. Denote this optimal solution as $w^* $. Using Lagrange multipliers, we minimize the following function (omitted $s$ for clarity).

$$
S(w,\lambda) = \sum_{i,j} \Delta w_{ij}^2 + \sum_j \lambda_j \sum_i (\Delta w_{ij} a_i - w_{ij} \Delta a_i)
$$

where $$\Delta w_{ij} = w^*_{ij} - w_{ij}$$ and $\Delta a_i = a_i' - a_i$. Solving this optimization problem, we achieve

$$
\Delta w_{ij} = \frac{a_i}{ \vert a \vert ^2}\sum_{i'} \Delta a_{i'}w_{i'j}
$$

For training data $x_k$, we choose a test data $x_k'$ that is closest to $x_k$ (in Euclidean distance) and have the same label. We write the generalization gap as

$$
\Delta l_k \equiv l(x_k',w) - l(x_k,w) = l\left(x_k,w_k^* \right) - l(x_k,w) = g_k \cdot \Delta w_k = \sum^M_{n=1} g_{k,n}\Delta w_{k,n}
$$

where $g_k = \frac{\Delta l_k}{\vert \Delta w_k \vert^2} \Delta w_k$. Notation-wise, the subscript $k$ denotes the data, and depending on this data, a dual weight $w^*_k$ is deterministically chosen using the equation above. The variables $g_k$ and $\Delta w_k$ are components of $w_k$ and $\Delta w$ in an eigenbasis of the Hessian $\nabla^2 L(w)$ of the training loss function evaluated at $w$ (where $n$ represents the direction).

Empirical evidence (from fully connected neural networks trained on MNIST and CIFAR-10), the means $\mu_{g,n}$, $\mu_{w,n}$ are negligible compared to the standard deviations $\sigma_{g,n}$, $\sigma_{w,n}$, and the mutual correlation coefficient $c_n$ is roughly independent of the direction $n$. Therefore,

$$
\Delta l(w) \approx \sum^M_{n=1} c_n \sigma_{w,n} \sigma_{g,n}
$$

Now, we observe that $\sigma_{w,n}$ corresponds to the weight norm and $\sigma_{g,n}$ corresponds to the sharpness of the loss function. Therefore, the generalization gap is codetermined by these two factors.


## References
[1] Dinh, Laurent, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. 2017. “Sharp Minima Can Generalize for Deep Nets.” arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/1703.04933. \
[2] Feng, Yu, Wei Zhang, and Yuhai Tu. 2023. “Activity–Weight Duality in Feed-Forward Neural Networks Reveals Two Co-Determinants for Generalization.” Nature Machine Intelligence 5 (8): 908–18.
