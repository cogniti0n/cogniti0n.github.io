---
title: Policy/Value Iteration, Model-Free Learning
date: 2025-07-29 20:00:00 +0800
description: Finding the solution to Bellman optimality equations.
categories: [reinforcement-learning]
tags: [rl]
math: true
toc: false
---

$$
    \def\argmin{\mathop{\mathrm{argmin}}}
    \def\argmax{\mathop{\mathrm{argmax}}}
    \def\expectation{\mathop{\mathbb{E}}}
    \def\SS{\mathcal{S}}
    \def\PP{\mathcal{P}}
    \def\AA{\mathcal{A}}
$$

## Policy Iteration

Policy/value iteration are forms of **dynamic programming**. They are effective when the total MDP $(\SS, \AA, \PP, R, \gamma)$ is known and $\SS$ and $\AA$ are small in size. 

Recall the state-value function's inductive formula.

$$
v^{\pi}(s) = R^\pi(s) + \gamma \sum_{s' \in \SS} \PP^\pi (s,s') \cdot v^\pi(s')
$$

$$
v^\pi = R^\pi + \gamma P^\pi v^\pi
$$

We iteratively find the solution of this self-consistant equation iteratively.

> **Algorithm** (Iterative policy evaluation) Given a policy $\pi$,
> - $v_1(s) = \text{ a random value }$
> - $v_{k+1}(s) = R^\pi(s) + \gamma \sum_{s' \in \SS} \PP^\pi(s,s') \cdot v_k(s')$
> Then by the contraction mapping theorem, $\lim_{k \to \infty} v_k = v^\pi$ converges.

Now that we can find $v^\pi$ when given a policy $\pi$, we need to iteratively find the optimal policy $\pi^* $. 
- Start from any initial policy $\pi_1$.
- Given a policy $\pi_l$, find the state-value function $v^{\pi_l}$. Then calculate the action-value function by using

$$
q^{\pi_l} (s,a) = R(s,a) + \gamma \sum_{s' \in \SS} \left( \PP(s,a,s') \cdot v^{\pi_l}(s') \right)
$$

- Find a 'better' policy $\pi_{l+1} = \text{greedy}(v^{\pi_l})$, i.e.

$$
\pi_{l+1}(s) = \argmax_{a \in \AA} \, q^{\pi_l}(s,a)
$$

- Now the question remains - does $\pi_l$ converge? The answer turns out to be yes.

> **Theorem** $\lim_{l \to \infty} \pi_l = \pi^* $

_sketch of proof_. 

- $\pi_l : \SS \to \AA$, then $\pi_{l+1}(s) = \argmax_{a \in \AA} \, q^{\pi_l}(s,a)$ where

$$
q^{\pi_l}(s,a) = R(s,a) + \gamma \sum_{s' \in \SS} (\PP(s,a,s') \cdot v^{\pi_l}(s'))
$$

- Then, $$q^{\pi_l}(s, \pi_l(s)) = \max_{a \in \AA} q^{\pi_l}(s,a) \ge q^{\pi_l}(s, \pi_l(s)) = v^{\pi_l}(s)$$

- Therefore, $v^{\pi_l}(s) \le q^{\pi_l}(s,\pi_{l+1}(s)) \le v^{\pi_{l+1}}(s)$

- If $v^{\pi_l}(s) = v^{\pi_{l+1}}(s)$, then $v^{\pi_l}(s) = \max_{a \in \AA} q^{\pi_l}(s,a)$.

- Thus, the Bellman optimality equation is satisfied, and so $\pi_l$ is an optimal policy.

## Value Iteration

Value iteration does not compute the policy $\pi$ and instead iteratively calculates the optimal value function. Recall the self-consistant Bellman optimality equations.

$$
v^* (s) = \max_{a \in \AA} \left( R(s,a) + \gamma \sum_{s' \in \SS} \PP(s,a,s') \cdot v^* (s') \right)
$$

> **Algorithm** (Iterative evaluation of optimal value function) 
> - First pick a random values for $v_1(s)$
> - Then, iteratively calculate
>
> $$
v_{k+1}(s) = \max_{a \in \AA} \left( R(s,a) + \gamma \sum_{s' \in \SS} \PP(s,a,s') \cdot v_k(s') \right)
> $$

> **Theorem** $\lim_{k \to \infty} v_k = v^* $ converges.

After $v^* $ is calculated, we can set the optimal policy $\pi^* = \text{greedy}(v^* )$. 

## Model-Free Learning

When we do not know the full MDP, we use different methods to calculate the optimal policy.