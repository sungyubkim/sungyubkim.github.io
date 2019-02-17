---
layout: post
title: A note on regularizing GAN
date: 2019-2-17
tags:
  - GAN
mathjax: true
---



## 1. Vanilla GAN

If D is optimal critic(or discriminator) and G can improve its loss, then
$$
p_g \rightarrow p_{data}.
$$
$\square$

- How about G cannot improve its loss? (like gradient vanishing)

- Optimal critic is defined as
  $$
  D_{G}^{\star}(x) = \frac{p_r(x)}{p_r(x) + p_g(x)}.
  $$
  $\square$

  The density of each sample $x$ is hard to estimate in the case of high-dim (like image, text, etc.)

  