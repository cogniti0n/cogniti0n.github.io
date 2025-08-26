---
title: Basics of Monte-Carlo
date: 2025-07-23 11:33:00 +0800
description: Basics and methods in MC / Markov-chain Monte Carlo
categories: [math-statistics]
tags: [lecture-notes]
math: true
toc: false
---

$$
    \def\argmin{\mathop{\mathrm{argmin}}}
    \def\argmax{\mathop{\mathrm{argmax}}}
    \def\expectation{\mathop{\mathbb{E}}}
    \def\R{\mathcal{R}}
$$

## Monte Carlo Method

Monte-carlo methods use repeated random sampling to perform tasks such as estimating a variable, or sampling from an arbitrary random distribution (_Markov-chain Monte Carlo_). First we look at Monte Carlo estimation the expectation value of a function. To estimate $I = \expectation_{X \sim p} \left[ \phi(X) \right]$, we sample i.i.d. distributed random variables $X_1, \dots, X_n \sim p$ and use $\hat{I} = \frac{1}{n}\sum^n_{k=1} \phi(X_k)$. By the law of large numbers, as $n \to \infty$, we can accurately estimate the desired outcome.

Below we state methods used to improve the accuracy and efficiency of Monte-Carlo sampling.

### Importance Sampling

Importance sampling (IC) is a technique for reducing the variance of a MC estimator. A key insight is to transform the expectation value using a proxy.

$$
I = \int \phi(x) p(x) \, dx = \int \frac{\phi(x) p(x)}{q(x)} q(x) \, dx = \expectation_{X \sim q} \left[ \frac{\phi(X) p(X)}{q(X)} \right]
$$

Then we use the following estimator instead of the original.

$$
\hat{I} = \frac{1}{N} \sum_i \phi(X_i) \frac{p(X_i)}{q(X_i)}
$$

The weight $p(x) / q(x)$ is called the _likelihood ratio_ or the _Radon-Nikodym_ derivative.

For examples, consider the setup of estimating $\mathrm{Pr}\left[ X > 3 \right] \simeq 0.00135$ where $X_i \sim \mathcal{N}(0,1)$. Then if we use the normal MC estimator

$$
\frac{1}{N} \sum_i \mathbf{1}_{\{ X_i > 3\}}
$$

the probability outcome can come out as $0$ without a sufficiently large amount of samples. To preven tthis, we use the IS estimator 

$$
\frac{1}{N}\sum_i \mathbf{1}_{\{ Y_i > 3 \}} \exp \left( \frac{(Y_i - 3)^2 - Y_i^2}{2} \right)
$$

where $Y_i \sim \mathcal{N}(3,1)$.

Quantitatively, we can verify the benefit of IS by meausring a decreased variance. We call $q$ the _importance_ or _sampling distribution_, and choosing it poorly can lead to decreased performance. Optimally, we want to choose $q$ so that the variane is minimum. The theoretical optimum can be reached when

$$
q(x) = \frac{\phi(x) p(x)}{I} = \frac{\phi(x) p(x)}{\int \phi(x) p(x) \, dx}
$$

However, in most cases, we do not know the integrand. Therefore, we instead consider the optimization problem

$$
\mathop{\text{minimize }}_{q \in \mathcal{Q}} D_{KL} \left[ q \, \middle\| \, \frac{\phi(x)p(x)}{I} \right]
$$

Such an optimization process does not require knowledge of $I$.

$$
\begin{aligned}
D_{KL} \left[ q \, \middle\| \, \frac{\phi(x)p(x)}{I} \right] &= \expectation_{X \sim q} \left[ \log\left( \frac{I q(X)}{\phi(X) p(X)} \right) \right] \\
&= \expectation_{X \sim q} \left[ \log\left( \frac{q(X)}{\phi(X) p(X)} \right) \right] + \log I \\
\end{aligned}
$$

In practice, $q$ is usually a neural net parameterized by $\theta$, and we minimize the objective function above using SGD.

### Log-Derivative Trick

Consider the general setup where we wish to solve

$$
\mathop{\text{minimize }}_{\theta \in \R} \expectation_{X \sim f_\theta} \left[\phi(X) \right]
$$

with SGD. When calculating the gradients, we see that

$$
\begin{aligned}
\nabla_\theta \int \phi(x) f_\theta(x) \, dx &= \int \phi(x) \nabla_\theta \, f_\theta(x) \, dx \\
                                             &= \int \phi(x) \frac{\nabla_\theta \, f_\theta(x)}{f_\theta(x)} f_\theta(x) \, dx \\
                                             &= \expectation_{X \sim f_\theta} \left[ \phi(X) \nabla_\theta \, \log f_\theta(X) \right]
\end{aligned}
$$

Therefore, we can treat $\phi(X) \nabla_\theta \, \log f_\theta(X)$ as the stochastic gradients of the loss function. This technique is called the _log-derivative_ trick. It is especially useful when dealing with exponential families. However, in practice the gradients have high variance and thus convergence is slow.

### Reparametrization Trick

When sampling from a Gaussian, we can reparametrize into random variables with the standard normal distribution, i.e.,

$$
\nabla_{\mu,\sigma} \, \expectation_{ X \sim \mathcal{N}(\mu, \sigma^2)}\left[ \phi(X) \right] = \expectation_{Y \sim \mathcal{N}(0,1)} \left[ \nabla_{\mu,\sigma}\phi(\mu + \sigma Y) \right] \simeq \frac{1}{B} \sum^B_{i=1} \phi'(\mu + \sigma Y_i) \begin{bmatrix} 1 \\ Y_i \end{bmatrix} \quad Y_i \sim \mathcal{N}(0,1)
$$

These gradients have smaller variane and thus SGD converges faster.

## Markov-Chain Monte Carlo

The goal of MCMC is to randomly sample from a given probability distribution $p$, which is a difficult task. To perform this, we construct a Markov process $x_t$ that follows $p$.

$$
p(x_t|x_{t-1},\dots,x_1)=p(x_t|x_{t-1})
$$

If we perform this process long enough, then the samples will reach a stationary state - hence, reversible. Therefore, 
$$
p(x_t, x_{t-1}) = p(x_t|x_{t-1})p(x_{t-1})=p(x_{t-1}|x_t)p(x_t)
$$
. This condition is called *detailed balance*.

We consider the simplest probability distribution, which is when

- If $p(x_t) < p(x_{t-1})$ → 
$$p(x_{t-1}|x_t) = 1$$ / $$p(x_t | x_{t-1}) = p(x_t) / p(x_{t-1})$$
- If $p(x_t) < p(x_{t-1})$ → 
$$p(x_{t}|x_{t-1}) = 1$$ / $$p(x_{t-1} | x_t) = p(x_{t-1}) / p(x_t)$$

Therefore, we end up with the *Metropolis algorithm.*

$$
p(x|x')=\mathrm{min}\left[1,\frac{p(x)}{p(x')}\right]
$$

>**Algorithm** (Markov-Chain Monte Carlo)
> - Take an arbitrary $x_1$.
> - Take $x'=x_1 + \xi$ where $\xi \sim \mathrm{Unif}[-0.01, 0.01]$ (or some small random number)
> - Accept $x'$ with the probability
>
>$$
p(x'|x_1)=\mathrm{min}\left[1,\frac{p(x')}{p(x_1)}\right]
>$$
>
> - If rejected, then take $x_2 = x_1$.
> - Repeat the process above.

If we drop the sample from the initial burning period ($ \sim 1000$), then we end up with samples following the probability distribution $p$.

## References
[1] Ryu, Ernest K. 2024. Lecture slides on "Mathematical Foundations of Deep Neural Networks, Spring 2024". \
[2] Jo, Junghyo. 2025. Lecture notes on "Computation, Learning, and Physics, Spring 2025".