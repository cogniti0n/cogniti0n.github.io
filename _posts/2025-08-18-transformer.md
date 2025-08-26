---
title: Attention is All You Need
date: 2025-08-18 05:50 +0800
description: Basic transformer architectures
categories: [machine-learning-deep-learning, deep-learning]
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

## Transformer Architectures

### Encoder Architecture

Consider a sequence of $N$ input vectors $$\{h_i\}^N_{i=1} \subset \R^{D}$$ written as an input matrix $\mathbf{H} = [h_1,\dots,h_N] \in \R^{D \times N}$ (each $h_i$ is a column of $\mathbf{H}$).

> **Definition** (Multi-head Attention) A self-attention layer with $M$ heads is defined as follows. First The model parameters are $$\theta = \{ (\mathbf{Q}_m, \mathbf{K}_m, \mathbf{V}_m) \}_{m=1}^M$$ where $\mathbf{Q}_m, \mathbf{K}_m \in \R^{d_k \times D}$ and $\mathbf{V}_m \in \R^{d_v \times D}$, and the output projection matrix $\mathbf{O} \in \R^{D \times Md_v}$. The output is then defined as
>
> $$
\tilde{\mathbf{H}} = \text{Attn}_{\theta}(\mathbf{H}) \equiv \mathbf{H} + \mathbf{O}
\begin{bmatrix}
 \text{head}_1 \\
 \vdots \\
 \text{head}_M
\end{bmatrix}
> $$
>
> Where each attention head is defined as
>
> $$
\text{head}_m = (\mathbf{V}_m \mathbf{H})\,\text{softmax}\left(\frac{(\mathbf{K_m} \mathbf{H})^T (\mathbf{Q_m} \mathbf{H})}{\sqrt{d_k}}\right)
> $$

After performing layer normalization, the output is passed through a standard MLP layer.

> **Definition** (MLP layer) An MLP layer with hidden dimension $D'$ has parameters $\theta = (\mathbf{W}_1, \mathbf{W}_2)$
>
> and is defined as
>
> $$
\tilde{\mathbf{H}} = \mathrm{MLP}_\theta(\mathbf{H}) = \mathbf{H} + \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{H})
> $$
>
> Where $\sigma$ is the standard $\mathrm{ReLU}$ activation.

Finally, layer normalization is performed, and the output is cached to serve as the value and key inputs to the decoder multi-head attention layer.

The process of adding $\mathbf{H}$ with the transformed attention output stems from the idea of a _residual connection_, which is the core idea of the ResNet CNN architecture.

### Decoder Architecture

We define a _masked_ multi-head attention. This stems from the fact that guessing the next word is easy if the information about the next word is already given. Therefore, we must 'mask the future' to effectively train the model. By setting all 'future values' with $-\infty$, the softmax function effectively drops to 0.
