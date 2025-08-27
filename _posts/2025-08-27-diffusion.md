---
title: Diffusion Models, DDPM, DDIM
date: 2025-08-25 05:50 +0800
description: Introduction to diffusion models.
categories: [machine-learning-deep-learning, unsupervised-learning]
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

## Diffusion Models

The core idea of diffusion models is based on the Langevin equation. Start from a data vector (e.g. the pixels to an MNIST data), then for each step, add a random Gaussian noise. When the iteration number is sufficiently large enough, the data becomes a completely uncorrelated noise. Then we start from uncorrelated noise and return back to the original by removing noise layers, and recover a randomly generated data.

Denote the data at each step as $x_0, x_1, \dots$, and randomly sampled noise vectors as $\varepsilon_t \sim \mathcal{N}(0,1)$. Each step, choose a parameter $\alpha_t \in [0,1]$ and apply the transformation

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \varepsilon_t
$$

Then by definition the sequence $x_t$ is Markovian. Denote the probability distribution as $q$.

$$
q(x_t \, | \, x_{t-1},x_0) \propto \exp \left[ - \frac{(x_t - \sqrt{\alpha_t}x_{t-1})}{2(1-\alpha_t)} \right]
$$

A nice property holds for this sequence:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \bar{\varepsilon}_t
$$

where $\bar{\alpha}_t = \alpha_1 \alpha_2 \cdots \alpha_t$ and $\bar{\varepsilon}_t \sim \mathcal{N}(0,1)$. This property can easily be verified using induction. Then, the probability distribution solely relies on the initial value $x_0$.

$$
q(x_t \, | \, x_0) \propto \exp \left[-\frac{(x_t - \sqrt{\bar{\alpha}_t}x_0)^2}{2(1-\bar{\alpha}_t)}\right]
$$

Then to propagate backwards, we need to find the reverse probability distribution

$$
q(x_{t-1}\,|\,x_t,x_0) = \frac{q(x_t \, | \, x_{t-1},x_0) \cdot q(x_{t-1} \, | \, x_0)}{q(x_t \, | \, x_0)} \sim \mathcal{N}(\mu_q(x_t, x_0, t), \sigma_q^2(t))
$$

Since the distributions $q$ are all Gaussian, the reverse distribution is also Gaussian. The mean and variance can be explicitly calculated.

$$
\mu_q(x_t,x_0,t) = \frac{\sqrt{\alpha_t} (1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{(1-\bar{\alpha}) \sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} x_0
$$

$$
\sigma_q^2(t) = \frac{(1-\alpha_t)(1-\bar{\alpha}_t)}{1-\bar{\alpha}_t}
$$

The main problem we face is that the distribution contains the inital value $x_0$. A generative model's goal is to generate a completely new result, without knowing the initial $x_0$. Therefore, we sample from a proxy distribution

$$
p(x_{t-1}\,|\,x_t) \sim \mathcal{N}(\mu_p(x_t,t), \sigma_p^2(t))
$$

That approximates the original distribution. Since $\sigma_q^2(t)$ is independent of $x_0$, we can set $\sigma_p(t) = \sigma_q(t)$. Then our goal becomes to find $\mu_p(x_t,t) \approx \mu_q(x_t,x_0,t)$. Consider a sample $x_0$ generated from this reverse process. Then define the log-likelihood

$$
\int dx_0 \, q(x_0) \log p(x_0)
$$

Our goal is to maximize this value. First observe that, from the Markovian property,

$$
\begin{aligned}
p(x_0) &= \int dx_1 \cdots dx_T \, p(x_0, \dots, x_T) \\
       &= \int dx_1 \cdots dx_T \, p(x_T) \prod^T_{t=1} p(x_{t-1}\,|\,x_t) \frac{q(x_1,\dots,x_T \, | \, x_0)}{\prod^T_{t=1} q(x_t \, | \, x_{t-1})} \\
       &= \int dx_1 \cdots dx_T \, q(x_1, \dots, x_T \, | \, x_0) p(x_T) \prod^T_{t=1} \frac{p(x_{t-1}\,|\,x_t)}{q(x_t\,|\,x_{t-1})}
\end{aligned}
$$

Then use Jensen's inequality to show

$$
\int dx_0 \, q(x_0) \log p(x_0) \ge \int dx_0 \cdots dx_T \, q(x_0, \dots, x_T) \left[ \log p(x_T) + \sum^T_{t=1} \log \frac{p(x_{t-1}\,|\,x_t)}{q(x_t\,|\,x_{t-1})} \right]
$$

Using the same method as the derivation to the ELBO inequality, define

$$
L_{t-1} = \int dx_0 dx_t \, q(x_0,x_t) \, D_{KL}\left[ q(x_{t-1}\,|\,x_t,x_0) \middle\| p(x_{t-1}\,|\,x_t) \right]
$$

Then the right hand side of the inequality becomes

$$
\int dx_T \, p(x_T) \log p(x_T) - \sum^T_{t=1} L_{t-1} - \int dx_0 dx_T \, q(x_0,x_T) \log q(x_T \, | \, x_0)
$$

Just like variational inference, we maximize the quantity $\sum^T_{t=1} L_{t-1}$. Since $q$ and $p$ are Gaussian, the KL divergence can be explicitly calculated

$$
D_{KL}\left[ q(x_{t-1}\,|\,x_t,x_0) \middle\| p(x_{t-1}\,|\,x_t) \right] = \frac{1}{2\sigma^2_q(t)} \| \mu_p - \mu_q \|^2
$$

A neural network is used to model $\mu_p(x_t,t ; \theta)$. During training time, inputs $x_t,t$ and therefore the 'goal' $\mu_q(x_t,x_0,t)$ can be explicitly constructed. Therefore, using traditional deep learning methods we construct $\mu_p(x_t,t) \approx \mu_q(x_t,x_0,t)$. Once we construct $\mu_p$ for each time step, we reverse the process and sample

$$
x_{t-1} = \sqrt{\alpha_t} \mu_p (x_t,t) + \sqrt{1-\alpha_t} \varepsilon
$$

where $\varepsilon \sim \mathcal{N}(0,1)$.

The problem with this approach is apparent - for every time step, we have to train a brand new neural network and therefor is extremely computationally expensive.

## Denoising Diffusion Probabilistic Model (DDPM)

DDPM uses several simple tricks to make the process more efficient. First, we do not have to train the whole $\mu_p$ for each time step. Observe that

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} - \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}}\bar{\varepsilon}_t
$$

Now use

$$
\mu_q(x_t,x_0,t) = \frac{\sqrt{\alpha_t} (1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{(1-\bar{\alpha}) \sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} x_0 = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t} \sqrt{\alpha_t}} \bar{\varepsilon}_t
$$

To reduce computational cost, reparametrize $\mu_p$ so that

$$
\mu_p(x_t,t) = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t} \sqrt{\alpha_t}} \hat{\varepsilon}_t
$$

Then the KL divergence term becomes

$$
D_{KL}\left[ q(x_{t-1}\,|\,x_t,x_0) \middle\| p(x_{t-1}\,|\,x_t) \right] = \frac{1}{2\sigma_q^2(t)} \frac{(1-\alpha_t)^2}{(1-\bar{\alpha}_t)\alpha_t} \| \bar{\varepsilon}_t - \hat{\varepsilon}(x_t) \|^2
$$

The total objective function becomes

$$
L = \sum^T_{t=1} L_{t-1} = \sum^T_{t=1} \int dx_0 \, dx_t \, q(x_0,x_t) \frac{1}{2\sigma_q^2(t)} \frac{(1-\alpha_t)^2}{(1-\bar{\alpha}_t) \alpha_t} \| \bar{\varepsilon}_t - \hat{\varepsilon}(x_t) \|^2
$$

The DDPM model simplifies this equation even more by ignoring the weight terms

$$
L_{\mathrm{simple}} = \expectation_{t \sim [1,T],x_0,\varepsilon_t} \left[ \| \bar{\varepsilon}_t - \hat{\varepsilon}(x_t) \|^2 \right]
$$

which was shown to perform better empirically. The final idea used in the DDPM model is to use a U-Net to find $\hat{\varepsilon}_t$, which was shown to improve accuracy.

## References

[1] Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In arXiv [cs.LG]. arXiv. \
[2] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In arXiv [cs.LG]. arXiv.
