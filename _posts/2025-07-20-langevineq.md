---
title: Langevin Equation for Brownian Motions
date: 2025-07-20 20:00:00 +0800
description: Solving a simple Langevin equation for Brownian motions.
categories: [physics, statistical-physics]
tags: [langevin-equation]
math: true
toc: false
---
## Solving The Langevin Equation

The Langevin equation is a stochastic differential equation of the form

$$
m\frac{dv}{dt} = F_{\mathrm{ext}} - \gamma v + \eta
$$

where $\langle \eta(t) \rangle = 0$ and $\langle \eta(t) \eta(t') \rangle = A \delta(t-t')$. We look at the condition where $F_{\mathrm{ext}}=0$. Then, from statistical mechanics we can infer that at the equilibrium state, the particles will have a Boltzmann distribution.

$$
P_{\mathrm{eq}}(v)\propto \exp\left(-\frac{mv^2}{2kT} \right)
$$

We will later see that this indeed is true.

First we look at the cumulant generating function

$$
\phi(k,t) = \log\left\langle e^{kv(t)} \right\rangle
$$

and denote the $n$-th cumulant as $K_n(t)$. Now consider an infinitesimal time difference $\Delta t$

$$
\begin{aligned}
\phi(k,t+\Delta t) &= \log \left\langle e^{kv(t)} \exp\left[ k \int^{t+\Delta t}_{t} dt' \left[-\frac{\gamma}{m}v(t')+\frac{1}{m}\eta(t') \right] \right] \right\rangle \\ &\simeq \log \left\langle e^{kv(t)}\left[1 - \frac{k\gamma}{m}v(t)\Delta t+k\int dt'\,\frac{1}{m}\eta(t')+\frac{k^2}{2m}\int\int dt'dt''\,\eta(t')\eta(t'') \right] \right\rangle
\end{aligned}
$$

Since $\Delta t$ is infinitesimal, we can replace the $\eta$ terms with the mean values.

$$
\begin{aligned}
\phi(k,t+\Delta t) &\simeq \log \left\langle e^{kv(t)}\left[1-\frac{k\gamma}{m}v(t)\Delta t + \frac{k^2}{2m}A\Delta t\right] \right\rangle \\ 
&= \log\left[\left\langle e^{kv(t)}\right\rangle - \frac{k\gamma}{m}\left\langle v(t) e^{kv(t)}\right\rangle \Delta t + \frac{k^2 A}{2m}\left\langle e^{kv(t)} \right\rangle\Delta t\right] \\ 
&\simeq \phi(k,t) - \frac{k\gamma}{m}\frac{\left\langle v(t)e^{kv(t)} \right\rangle}{\left\langle e^{kv(t)} \right\rangle} \Delta t + \frac{k^2A}{2m^2}\Delta t = \phi(k,t) - \frac{k\gamma}{m}\frac{\partial \phi}{\partial k} \Delta t + \frac{k^2 A}{2m^2} \Delta t
\end{aligned}
$$

As $\Delta t \to 0$, we can see that

$$
\frac{\partial \phi}{\partial t} = -\frac{k\gamma}{m}\frac{\partial \phi}{\partial k} + \frac{k^2A}{2m}
$$

Therefore, in terms of the cumulants,

$$
\dot{K}_n=-\frac{n\gamma}{m}K_n + \frac{A}{m^2} \delta_{n,2}
$$

i.e., all the other cumulants decay exponentially except for the variance. As $t \to \infty$, $K_2 \to \frac{A}{2\gamma m}$ and the distribution of $v$ reaches a zero-mean Gaussian, which matches with the Boltzmann distribution. Since

$$
P_{\mathrm{eq}} \propto \exp\left(-\frac{\gamma m v^2}{A}\right)
$$

we see that the constant $A = 2\gamma k_B T$.

## Steady-State Variables

Now we look at some steady-state variables.

#### Velocity Correlation

Hereinafter assume $t > t'$. Then,

$$
\frac{d}{dt}\left\langle v(t)v(t') \right\rangle = - \frac{\gamma}{m}\left\langle v(t)v(t')\right\rangle + \frac{1}{m} \left\langle \eta(t) v(t')\right\rangle
$$

The second term is zero since $\eta(t)$ and $v(t')$ can be separated. Therefore,

$$
C_{\mathrm{eq}}(t)\equiv\lim_{t' \to \infty}\left\langle v(t+t')v(t') \right\rangle = \left\langle v^2 \right\rangle _{\mathrm{eq}} e^{-\frac{\gamma t}{m}} = \frac{k_BT}{m} e^{-\frac{\gamma t}{m}}
$$

Therefore, the correlation decays exponentially with time.

#### Mean Square Displacement

It is trivial that the mean displacement is zero. Then let’s look at the mean square displacement $\left\langle x(t) \right\rangle^2_{\mathrm{eq}}$.

$$
\begin{aligned}
\left\langle x(t) \right\rangle^2_{\mathrm{eq}} &= \int^t_0dt'\int^t_0dt''\,\langle v(t')v(t'')\rangle_{\mathrm{eq}} \\
&= 2 \int^t_0 dt'' \int^t_{t''} dt'\, C_{\mathrm{eq}}(t'-t'') \\ 
&= 2\int^t_0 dt'' \int^{t-t''}_0 ds \, C_{\mathrm{eq}}(s) \\
&= 2\int^t_0 ds \int^{t-s}_0 dt'' \, C_{\mathrm {eq}}(s)\\ 
&= 2 \int^t_0 ds \,(t-s)C_{\mathrm{eq}}(s)
\end{aligned}
$$

(where we restrict $t' > t''$ and multiply a factor of 2) If we calculate using the velocity correlation from before,

$$
\left\langle x(t) \right\rangle^2_{\mathrm{eq}} = \frac{2k_BT}{\gamma}\left[t-\frac{m}{\gamma}\left(1-e^{-\frac{\gamma}{m}t}\right)\right] = \begin{cases} \frac{k_BT}{m}t^2 & \text{if }t\ll\frac{m}{\gamma} \\ \frac{2k_BT}{\gamma}t & \text{if }t \gg \frac{m}{\gamma}\end{cases}
$$

Where the first case is called *ballistic*, i.e., the displacement $\propto t^2$ and the second case is called *diffusive*. We can define a diffusion constant

$$
D \equiv \lim_{t \to \infty}\frac{\left\langle x(t) \right\rangle^2_{\mathrm{eq}}}{2t}=\frac{k_BT}{\gamma}
$$