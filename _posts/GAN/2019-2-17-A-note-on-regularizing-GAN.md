---
layout: post
title: A note on regularizing GAN(to be updated)
date: 2019-2-17
tags:
  - GAN
mathjax: true
---



## 1. Vanilla GAN

- NIPS 2014

- If D is optimal critic(or discriminator) and G can improve its loss, then

$$
p_g \rightarrow p_{data}.
$$

​	$\square$ 

​	How about G cannot improve its loss? (like gradient vanishing)

- Optimal critic is defined as
  $$
  D_{G}^{\star}(x) = \frac{p_r(x)}{p_r(x) + p_g(x)}.
  $$
  

  $\square​$

  The density of each sample $x$ is hard to estimate in the case of high-dim (like image, text, etc.)

  

## 2. Toward principled methods for training GANs

- ICLR 2017
- There exists a optimal critic whose gradient on both real data/fake data are 0. (So there is no improvement on G by alternative gradient descent(AGD) in this case. Therefore, there is no need to be $p_g \rightarrow p_d$. )
- If we mix noise (like additive noise) to real/fake data, then we can make lower bound of gradient on data.



## 3. WGAN

- ICML 2017

- Vanilla GAN use Jensen-Shannon distance. How about change the discrepancy measure between real/fake data?

- The Wasserstein distance in GAN is continuous everywhere and differentiable almost everywhere. (So we can prevent gradient vanishing problem in GAN.)

  

## 4. WGAN with Gradient Penalty(GP)

- NIPS 2017
- WGAN with weight clipping(Vanilla WGAN) is sensitive to its clipping criteria.
- So use Gradient Penalty between mixture of data ($\tilde{x}$), which doen not suffer from the same problem.
- This Gradient Penalty derived from the optimal critic function in Optimal Transportation Theory.



## 5. Consistency-term(CT) GAN

- ICLR 2018
- Mixing only one data from each distribution is not enough for regularization.
- Consider additional Consistency-term by perturabating real data.



## 6. W-div GAN

- ECCV 2018
- The Lipschitz condition for WGAN is hard to satisfy.
- Define new divergence(Wasserstein divergence) by calculating zero-centered Gradient Penalty.



## 7. Variational Discriminator Bottleneck

- ICLR 2019
- Mixing fixed noise used in *Towards principled methods* is not good enough because of its negligible effects.
- We can also regularize discriminator by using Deep Variatioanl Information Bottleneck(DVIB) techniques. It also gives lower bound of gradient for real/fake data.



## 8. Stabilizing Training of GAN through Regularization

- NIPS 2017
- Noise induced regularization can be efficiently approximated by penalizing gradient.
- Also propose WGAN-GP like algorithm for f-GAN.



## 9. Spectral normalization for GANs

- ICLR 2018
- Also regularizing critic by using spectral norm.



## 10. Improving generalization and stability of GANs

- ICLR 2019 (Also workshop on ICML 2018)
- Point that critic in GAN suffers catastrophic forgetting problem.
- Zero-centered Gradient Penalty improves the generatlization of the discriminator by pushing it towards the optimal discriminator.



## 11. On the convergence and stability of GANs(DRAGAN)

- Rejected to ICLR 2018
- Interpret the Alternative Gardient Descent scheme in GAN for Follow-The-Regularized-Leader in Algorithmic Game Theory.
- Also use Gradient Penalty.