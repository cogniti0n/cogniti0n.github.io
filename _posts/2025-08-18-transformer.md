---
title: Attention is All You Need
date: 2025-08-18 05:50 +0800
description: Basic transformer architectures
categories: [machine-learning-deep-learning, deep-learning]
tags: [paper-reading]
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

Consider a sequence of $N$ input vectors $$\{h_i\}^N_{i=1} \subset \R^{D \times N}$$ written as an input matrix $\mathbf{H} = [h_1,\dots,h_N] \in \R^{D \times N}$ (each $h_i$ is a column of $\mathbf{H}$). Also denote the activation function $\sigma(\cdot) = \mathrm{ReLU}(\cdot)$.

> **Definition** (Self-attention) A self-attention layer with $M$ heads has parameters 
>
> $$
\theta = \{ (\mathbf{V}_m,\mathbf{Q}_m,\mathbf{K}_m) \}_{m \in [M]} \subset \R^{D \times D}
> $$
>
> and is defined as
>
> $$
\tilde{\mathbf{H}} = \mathrm{Attn}_{\theta}(\mathbf{H}) = \mathbf{H} + \frac{1}{N} \sum^M_{m=1} (\mathbf{V}_m \mathbf{H}) \times \sigma\left( (\mathbf{Q}_m\mathbf{H})^\top (\mathbf{K}_m\mathbf{H}) \right) \in \R^{D \times N}
> $$
>
> In vector form, each of the tokens $h_i$ are transformed as
>
> $$
\tilde{h}_i = [\mathrm{Attn}_\theta (\mathbf{H})]_i = h_i + \sum^M_{i=1} \frac{1}{N} \sum^N_{j=1} \sigma\left( \langle \mathbf{Q}_m h_i, \mathbf{K}_m h_j \rangle \right) \cdot \mathbf{V}_m h_j
> $$

Instead of the normal ReLU, we use a normalized version $\sigma(\cdot)/N$ to make the attention weights sum to $O(1)$. 

> **Definition** (MLP layer) An MLP layer with hidden dimension $D'$ has parameters 
> 
> $$
\theta = (\mathbf{W}_1, \mathbf{W}_2) \in \R^{D' \times D} \times \R^{D \times D'}
>
> $$ 
>
> and is defined as
>
> $$
\tilde{\mathbf{H}} = \mathrm{MLP}_\theta(\mathbf{H}) = \mathbf{H} + \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{H})
> $$

A transformer is a composition of attention layers, followed by an MLP layer. 

## Encoder-Decoder Architectures Using Transformers

