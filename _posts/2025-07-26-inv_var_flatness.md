---
title: Inverse Variance-Flatness Relation in SGD
date: 2025-07-26 20:00:00 +0800
description: Summary of Feng, et al. 2021. Proc. Natl. Acad. Sci. U. S. A. 118 (9)
categories: [machine-learning-deep-learning, paper-reading]
tags: [paper-reading]
math: true
toc: false
---

$$
\def\T{\mathsf{T}}
$$

## Background

There is evidence in support of the notion that generalizable solutions exist at the flat minima of the loss function, but it is still unclear how SGD can find these minima. Regular GD is prone to being trapped by local minima and saddle points of the loss landscape. Adding isotropic noise can help GD (~ a Langevin-type equation) escape these minima, but cannot help with generalizability. This is because a 'useful noise' should depend on the loss landscape - in particular, the noise should be larger when the landscape is rougher and smaller when it is flatter. This can approximately be achieved by SGD.

In general, SGD dynamics have a fast initial learning period, and a slow 'exploration' phase when the training error reacher near 0 but the test error is still decreasing. To study the weight dynamics, the authors perform PCA and study the dynamics of the principal components.

## Inverse Variance-Flatness Relation

The neural network has two hidden layers of dimension $50$ each and trained on MNIST data. The center $ 50 \times 50 $ is analyzed. Take a large time window $T$ (~ 10 epochs) then the weight dynamics become

$$
w(t) = \langle w \rangle_T + \sum^N_{i=1} \theta_i \mathbf{p}_i
$$

The main variable is the variance of these components: $\sigma_i^2 \equiv T^{-1} \int^{t_0+T}_{t_0} \theta_i^2(t)\, dt$. There is a gap between the maximum variance and the second largest - which represents a dominant drifting component. The other $\sigma_i$ decay quickly with $i$, meaning that the dynamics are concentrated in a small number of PCA directions. The component $\mathbf{p}_1$ is highly aligned with $w_0 = \langle w \rangle_T$, therefore amplifying the weights. This may improve generalization by increasing the margin.

Let $w_0$ denote a solution found by SGD. The loss landscape around $w_0$ is calculated in the direction of the principal components, i.e.,

$$
L_i(\delta \theta) \equiv L(w_0 + \delta \theta p_i)
$$

Then, empirically, the landscape becomes flatter as the index increases. The authors quantify flatness by finding $\theta_i^l$ and $\theta_i^r$ such that $L_i(\theta^l) = L_i(\theta^r) = eL_0$ where $L_0 = L(w_0)$. Then the flatness parameter is defined as $F_i \equiv \theta^r_i - \theta^l_i$. Regardless of minibatch size and learning rate, the relation between the variance and flatness is an inverse power-law, i.e.,

$$
\sigma_i^2 \propto F_i^{-\psi}
$$

where $\psi \sim 4$ for MNIST data. Such a relation is different from the expectations of equilibrum statistical mechanics. In an equilibrium system with state variables $\theta$ and a free energy function $L(\theta)$, it is expected that $P(\theta) \propto \exp[-L(\theta)/T]$. By expanding the loss function to the second order, 

$$
L = L_{\mathrm{min}}\left( 1 + \sum_i \frac{(\theta \cdot p_i)^2}{F_i^2} \right)
$$

we expect $\sigma_i \propto TF_i^2$.

## SGD Dynamics and Search For Minima

Taking the continuous-time approximation, SGD can be described by

$$
\dot{\textbf{w}} = -\alpha \frac{\partial L^\mu}{\partial \textbf{w}} = - \alpha \frac{\partial L}{\partial \textbf{w}} + \eta(\textbf{w})
$$

The 'noise' $\eta = - \alpha \nabla_{\textbf{w}} \delta L^\mu$, which has zero mean. An important feature is that the noise is dependent on $\theta$. The strength of this noise along the $i$-th PCA direction is characterized by the 'temperature':

$$
T_i(\delta \theta , t) \equiv \frac{\alpha}{2} \left\langle \left\| \frac{\partial \delta L^\mu (w_0 + \delta \theta p_i)^2}{\partial \delta \theta} \right\| \right\rangle
$$

Then this temperature is large in the direction of sharper (and thus strong active learning) PCA component direction.

## Conclusion

- Around each solution, the loss landscape is flat in most PCA directions with only a small number of relevant directions where the loss landscape is sharp.
- The complexity of the solution found by SGD does not increase with the number of parameters, and the solution remains “simple” with good generalization performance in the overparameterized regime.
- SGD searches only in a small subspace for solu- tions after the initial transient and the dimension of the search space has only weak dependence on the network size in the overparameterized regime.

## References
[1] Feng, Yu, and Yuhai Tu. 2021. “The Inverse Variance-Flatness Relation in Stochastic Gradient Descent Is Critical for Finding Flat Minima.” Proceedings of the National Academy of Sciences of the United States of America 118 (9): e2015617118.