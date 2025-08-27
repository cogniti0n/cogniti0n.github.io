---
title: Policy/Value Iteration, Model-Free Learning
date: 2025-07-29 20:00:00 +0800
description: Finding the solution to Bellman optimality equations.
categories: [machine-learning-deep-learning, reinforcement-learning]
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
>
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
>
> - First pick a random values for $v_1(s)$
> - Then, iteratively calculate
>
> $$
v_{k+1}(s) = \max_{a \in \AA} \left( R(s,a) + \gamma \sum_{s' \in \SS} \PP(s,a,s') \cdot v_k(s') \right)
> $$

> **Theorem** $\lim_{k \to \infty} v_k = v^* $ converges.

After $v^* $ is calculated, we can set the optimal policy $\pi^* = \text{greedy}(v^* )$.

## Model-Free Learning

When we do not know the full MDP, we use different methods to calculate the optimal policy. In general, there are two methods - Monte-Carlo and temporal-difference.

### Monte-Carlo Learning

Recall the policy iteration algorithm. We iterate $v_k$ and choose the optimal policy $\pi_{l+1}(s) = \argmax_{a \in \AA} \, q^{\pi_l}(s,a)$. Monte-Carlo learning is similar in that we evaluate $q^{\pi_l}$ for a given policy $\pi_l$, then using the results, we improve the policy to $\pi_{l+1}$ - such a sequence eventually converges to the optimal policy $\pi^* $.

**Evaluating $q^{\pi_l}$**: Using the given policy $\pi_l$, we 'propagate' an initial state $s_0$ until it reaches termination. During this propagation, we average the return values for each $(\text{state},\text{action})$ pair (including discount) to find $q(s,a)$ for such a pair. We can implement either the first-visit MC method or the every-visit MC method, depending on whether we average over only the first visit or every visit of each $(\text{state},\text{action})$ pairs in a single episode.

**Policy Improvement**: Policy improvement is performed $\varepsilon$-greedy, i.e.,

$$
\pi_{l+1}(s,a) \gets \begin{cases} \varepsilon/|\AA| + 1 - \varepsilon & \text{if } a = \argmax_{a' \in \AA} \, q^{\pi_l}(s,a') \\ \varepsilon / |\AA| & \text{otherwise} \end{cases}
$$

### Temporal-Difference Learning

TD combines ideas from DP (policy/value iteration) and MC. Like MC, it learns directly from experience (model-free). Like DP, it updates the guess from previous guesses (without reaching the final state).

There are two main algorithms for TD learning - SARSA and Q-learning. The latter is well-suited to combine with classical supervised learning (using CNNs). In such cases, the model is called deep Q-learning.

### SARSA Learning

SARSA is an on-policy TD control algorithm. We iterate over $q$ and correspondingly set the policy $\varepsilon$-greedy. Given a step size $\alpha$, the main idea of SARSA is to improve the action-value function using the next $(\text{state},\text{action})$ pair, i.e.,

$$
q(s,a) \gets (1-\alpha) \cdot q(s,a) + \alpha \cdot \left( r + \gamma \cdot q(s',a') \right)
$$

The full algorithm is stated as follows.

> $\texttt{parameters}$: $\alpha \in (0,1]$, $\varepsilon > 0$ \
> $\texttt{initialize}$: $q(s,a) \gets \texttt{ random value for each } s \in \SS,\,  a \in \AA$ \
> $\texttt{for each episode}$ \
> $\quad s \gets \texttt{start state}$ \
> $\quad a \gets \pi(s) \texttt{ where } \pi = \varepsilon\textrm{-greedy}(q)$ \
> $\quad \texttt{for each step of the episode}$ \
> $\quad \quad \texttt{take action } a \texttt{ to get reward } r \texttt{ and next state } s'$ \
> $\quad \quad a' \gets \pi(s') \texttt{ where } \pi = \varepsilon\textrm{-greedy}(q)$ \
> $\quad \quad q(s,a) \gets q(s,a) + \alpha\cdot(r + \gamma \cdot q(s',a') - q(s,a)) $ \
> $\quad \quad s,a \gets s',a' $

### Q-Learning

Q-learning is off-policy algorithm, which means that we separate the moving policy with the iteratively learning policy. Instead of finding $a' = \pi(s')$, i.e., finding the next action value according to the given policy, we find $\max_{a' \in \AA} q(s',a')$. Therefore,

$$
q(s,a) \gets (1-\alpha) \cdot q(s,a) + \alpha \cdot \left( r + \gamma \cdot \max_{a' \in \AA} q(s',a') \right)
$$

The full algorithm is stated below.

> $\texttt{parameters}$: $\alpha \in (0,1]$, $\varepsilon > 0$ \
> $\texttt{initialize}$: $q(s,a) \gets \texttt{ random value for each } s \in \SS,\,  a \in \AA$ \
> $\texttt{for each episode}$ \
> $\quad s \gets \texttt{start state}$ \
> $\quad \texttt{for each step of the episode}$ \
> $\quad \quad a \gets \pi(s) \texttt{ where } \pi = \varepsilon\textrm{-greedy}(q)$ \
> $\quad \quad \texttt{take action } a \texttt{ to get reward } r \texttt{ and next state } s'$ \
> $\quad \quad q(s,a) \gets q(s,a) + \alpha\cdot(r + \gamma \cdot \max_{a' \in \AA} \cdot q(s',a') - q(s,a)) $ \
> $\quad \quad s \gets s' $

#### Monte-Carlo vs. Temporal-Difference

- Unlike MC, TD can learn without reaching the terminal state. MC has to reach the terminal state and get reward for each step. Therefore, TD can learn with incomplete episodes.
- TD learning has low-variance and high-bias, and is sensitive to initial values. MC learning has low-bias, and high-variance, and is insensitive to initial values.
- TD is effective on MDPs that satisfy the Markovian property well, and MC is effective on non-Markovian stochastic processes.

## References

[1] Silver, David. 2015. Lectures on Reinforcement Learning \
[2] Sutton, Richard S., and Andrew G. Barto. 2018. Reinforcement Learning: An Introduction. 2nd ed. Adaptive Computation and Machine Learning Series. Cambridge, MA: Bradford Books.
