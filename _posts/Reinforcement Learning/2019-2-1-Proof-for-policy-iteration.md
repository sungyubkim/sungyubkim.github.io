---
layout: post
title: Proof for policy iteration algorithm
date: 2019-2-9
tags:
  - Reinforcement Learning
mathjax: true
---



#### Step 1. Policy evalutation makes random value function converge to correct value function.

**Definition 1. $\gamma$-contraction mapping** 

 An operator $F$ on a normed vector space $\mathcal{X}$ is a $\gamma$-contraction, for $0<\gamma <1$, provided for all $x,y \in \mathcal{X}$


$$
\|F(x)- F(y)\| \le \gamma \|x-y\|
$$


$\square$



**Therem 1.  Contraction Mapping Theorem** 

For a $\gamma$-contraction $F$ in a complete noremed vector space $\mathcal{X}$, **$F$ converges to a unique fixed point** in $\mathcal{X}$.

$\square$



To use Contraction Mapping Theorem, remember **Bellman expectation equation, or consistency of fixed policy, ** is


$$
\begin{eqnarray}
	v_{\pi}(s) &= \sum_{a\in \mathcal{A}} \pi(a|s) \Big(\mathcal{R}_s^a + \gamma \sum_{s'\in \mathcal{S}} T_{s,s'}^a v_{\pi}(s')\Big)\\
    &= \sum_{a\in \mathcal{A}} \pi(a|s)\mathcal{R}_s^a + \gamma \sum_{s'\in \mathcal{S}}\sum_{a\in \mathcal{A}} \pi(a|s) T_{s,s'}^a v_{\pi}(s')
\end{eqnarray}
$$


If we define $r^{\pi}_s, T^{\pi}_{s,s'}$ as


$$
r_{s}^{\pi} \dot{=} \sum_{a\in \mathcal{A}} \pi(a|s)\mathcal{R}_s^a
$$

$$
T_{s,s'}^{\pi} = \sum_{a\in \mathcal{A}} \pi(a|s) T_{s,s'}^a
$$



Then, equation (3) can be written as following


$$
v_{\pi} = r^{\pi} + \gamma T^{\pi} v_{\pi}
$$


If we wefine column vector $F^{\pi}$ as 


$$
F^{\pi}(v) = r^{\pi} + \gamma T^{\pi} v
$$


, then we get followings


$$
\begin{eqnarray}
\|F^{\pi}(u) -  F^{\pi}(v)\|_{\infty} &= \|(r^{\pi} + \gamma T^{\pi} u )- (r^{\pi} + \gamma T^{\pi} v)\|_{\infty}\\
    &=\|\gamma T^{\pi} (u-v)\|_{\infty}\\
    &\le \gamma \|T^{\pi}\|_{\infty} \|(u-v)\|_{\infty} &\mbox{(Property of matrix $\infty$-norm)}\\
    &=\gamma \|(u-v)\|_{\infty} &\mbox{(Matrix $\infty$-norm:largest row sum)}
\end{eqnarray}
$$


So we get $F^{\pi}(v)$ is a $\gamma$-contraction mapping.

#### Step 2. Greed Policy Improvement makes current policy better.

Let $\pi$ is origianl policy and $\pi'$ is a new policy by applying greedy policy improvement. Then we get


$$
\begin{eqnarray}
v_{\pi}(s) &\le q_{\pi}(s,\pi'(s)) = \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_{\pi}(S_{t+1})|S_t = s] \\
    &\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_{\pi}(S_{t+1},\pi'(S_{t+1}))|S_t = s]\\
    &\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+1} + \gamma^2 q_{\pi}(S_{t+2},\pi'(S_{t+2}))|S_t = s] \\
    &\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \dots |S_t =s] = v_{\pi'}(s)
\end{eqnarray}
$$
