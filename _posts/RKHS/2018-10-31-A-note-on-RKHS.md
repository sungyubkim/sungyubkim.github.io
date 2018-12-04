---

layout: post
title: A Note on RKHS
date: 2018-10-31
tags: RKHS
mathjax: true

---

* **Remark** : This post is a rearrangement of [Percy Liang's statistical learning theory course 2016](https://web.stanford.edu/class/cs229t).

- [A Note on RKHS](#a-note-on-rkhs)
  * [What is Kernel?](#what-is-kernel-)
  * [Three views on RKHS](#three-views-on-rkhs)
    + [Predict, Experience and Backpropagate](#predict--experience-and-backpropagate)
  * [Feature map defines a kernel](#feature-map-defines-a-kernel)
  * [The Riesz Representation Theorem](#the-riesz-representation-theorem)
  * [Moore-Aronszajn Theorem](#moore-aronszajn-theorem)

# A Note on RKHS

## What is Kernel?

> **Definition** Kernel
>
> A function $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ is a **positive semidefinite (PSD) kernel** (or more simply, a **kernel**) if and only if for every finite set of points $x_1, \cdots, x_n \in \mathcal{X}$, the **kernel matrix** $K \in \mathbb{R}^{n \times n}$ defined by $K_{ij} = k(x_i, x_j)$ is positive semidefinite.

* Note: It is a generalization of positive semidefinite (PSD) matrix in the sense that *every finite principal sumbmatrix* of the kernel is a PSD matrix.

## Three views on RKHS

So from now on, we would like to understand this diagram completely!!!

<img src="img/2018-10-31-Notes-on-RKHS/three-view.png" height="300px">

* Feature map $(\phi : \mathcal{X} \rightarrow \mathbb{R}^d)$: Yeah, that **feature map** which we encounter in machine learning everyday.
* Kernel : We already define what kernel means mathematically. Although this is enough to progress our story, stop here and check what situation do we meet kernel in detailed setting.

### Predict, Experience and Backpropagate

Now assume that we **predict** the output $y \in \mathbb{R}$ by
$$
\begin{eqnarray}
\hat{y} = \big< w, \phi(x) \big>
\end{eqnarray}
$$
where $w \in \mathbb{R}^d$ is *weight vector*.

Now given answer of the predictions $y_1, \cdots, y_n$ for inputs $x_1, \cdots, x_n$ , our model **experience** loss by


$$
\begin{eqnarray}
L(w) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2}\big( y_i - \big< w, \phi(x_i) \big> \big)^2
\end{eqnarray}
$$


and the **gradient** of this loss is


$$
\begin{eqnarray}
\nabla_w L(w) &= \frac{1}{n} \sum_{i=1}^n (y_i - \big< w, \phi(x_i) \big>)\phi(x_i)\\
&= \frac{1}{n} \sum_{i=1}^n \alpha_i \phi(x_i)
\end{eqnarray}
$$


where $\alpha_i = y_i - \big< w, \phi(x_i) \big>$ is a kind of *prediction error*.

If our optimization algorithm use only gradient information, the *approximate optimal solution* would be


$$
\begin{eqnarray}
w^{OPT} = \sum_{t=1}^T\sum_{i=1}^n \alpha^t_i \phi(x_i)
\end{eqnarray}
$$



where $\alpha_i^t$ means prediction error of $i$-th data in $t$-th prediction.

So back to our *approximate optimal solution* the prediction can be denoted as


$$
\begin{eqnarray}
\hat{y} &= \big<w^{OPT}, \phi(x)\big>\\
&= \big< \sum_{t=1}^T\sum_{n=1}^n \alpha_i^t\phi(x_i), \phi(x)\big>\\
&= \sum_{t=1}^T\sum_{n=1}^n \alpha_i^t \big< \phi(x_i), \phi(x) \big>
\end{eqnarray}
$$


So if we define $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ as 


$$
\begin{eqnarray}
k(x_i,x_j) \dot{=} \big< \phi(x_i),\phi(x_j)\big>
\end{eqnarray}
$$


the equation (6) is 


$$
\begin{eqnarray}
\hat{y} =\sum_{t=1}^T\sum_{n=1}^n \alpha_i^t k(x_i, x).
\end{eqnarray}
$$

# Feature map defines a kernel

We have more general statement about equation (9) and (10).

> **Theorem** Feature map defines a kernel.
>
> If $\phi : \mathcal{X} \rightarrow \mathbb{R}^d$ is a feature map, then 
> $$
> \begin{eqnarray}
> k(x_i, x_j) \dot{=} \big< \phi(x_i), \phi(x_j) \big>
> \end{eqnarray}
> $$
> is a kernel.



> **proof**. Since kernel is determined by any principal submatrix, we only need to check arbitrary finite observation point $x_1, \cdots, x_n \in \mathcal{X}$. 
>
> If we define a kernel matrix $K$ as $K_{ij} = \big< \phi(x_i), \phi(x_j)\big>$, we have
> $$
> \begin{eqnarray}
> \alpha^{\top} K \alpha &= \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j \big<\phi(x_i), \phi(x_j) \big>\\
> &= \big<\sum_{i=1}^n \alpha_i\phi(x_i), \sum_{j=1}^n \alpha_j \phi(x_j) \big> \ge 0. \square
> \end{eqnarray}
> $$
>



Now back to the three views on RKHS, we define our main subject **Reproducing Kernel Hilbert Space (RKHS)**.



> **Definition** Hilbert space
>
> A Hilbert space $\mathcal{H}$ is an complete vector space with an inner product $\big<\cdot, \cdot, \big>: \mathcal{H} \times \mathcal{H} \rightarrow \mathbb{R}$ that satisfies the following properties
>
> 1. **(Symmetry)** $\big< x, y \big> = \big< y, x \big>$,  $\forall x, y \in \mathcal{H}$.
> 2. **(Linearity)** $\big<c_1x_1, c_2x_2, y\big> = c_1 \big< x_1, y\big> + c_2 \big< x_2, y\big>$,  $\forall x_1,x_2,y \in \mathcal{H}, \forall c_1, c_2 \in \mathbb{F}$.
> 3. **(Positive Definite)** $\big< x, x \big> \ge 0$ and equality holds only if $x = 0$.

* Note1: In here, the Hilbert space $\mathcal{H}$ usually means **good functions** whose domain is $\mathcal{X}$ and codomain is $\mathbb{R}^d$. We don't need to worry about what is good, but we can think our Hilbert space is $L^2$,in most time.
* Note2: The inner product of a Hilbert space $\mathcal{H}$ gives a natural norm $$\|x\|_{\mathcal{H}} = \sqrt{\big<x,x\big>}$$.
* Note3: The natural norm in Note1 gives a natural metric $$d_{\mathcal{H}}(x,y) = \|x-y\|_{\mathcal{H}}$$. So the complete vector space in above definition actually means *complete metric space*, any cauchy sequence converges. 

In Hilbert space, we call a function $L$ which maps an element of $\mathcal{H}$ to an element of  $\mathbb{R}$ **functional**.

Moreover in most time, we are interested in the functional  $L$ is linear, i.e. $L(cx + y)  = cL(x) + L(y), \forall x, y \in \mathcal{H}, \forall c \in \mathbb{R}$.

Now take an example of linear functional.

> **Definition** Evaluation functional
>
> Let $\mathcal{H}$ is an Hilbert space consisting of functions $f: \mathcal{X} \rightarrow \mathbb{R}$. For each $x \in \mathcal{X}$, we can define the **evaluation functional** $L_x : \mathcal{H} \rightarrow \mathbb{R}$ as
>
>
> $$
> \begin{eqnarray}
> L_x(f) \dot{=} f(x)
> \end{eqnarray}
> $$
>
>
> Note that the evaluation functional is a linear functional, i.e. $L_x(cf + g) = c L_x(f) + L_x(g), \forall f,g \in \mathcal{H}, \forall c \in \mathbb{R}$.



If a linear functional is **bounded**, we can do various analysis about that functional.

> **Definition** Bounded functional
>
> Given a Hilbert space $\mathcal{H}$, a functional $L : \mathcal{H} \rightarrow \mathbb{R}$ is bounded if and only if there exists an $M < \infty$ suct that 
>
>
> $$
> \begin{eqnarray}
> | L(f) | \le M \|f\|_{\mathcal{H}}, \forall f \in \mathcal{H}.
> \end{eqnarray}
> $$
>
>
>



* Note: If our functional $L$ is **linear**, the  **boundedness** is a necessary and sufficient condition of  **continuity** .

So our main subject **Reproducing Kernel Hilbert Space** (RKHS) demand that *any evaluation functional $L_x$ is a bounded linear functional.*

> **Definition** Reproducing Kernel Hilbert Space (RKHS)
>
> A Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}$ is a Hilbert space over functions $f : \mathcal{X} \rightarrow \mathbb{R}$ such that for each $x \in \mathcal{X}$, the evaluation functional $L_x$ is bounded.



# Moore-Aronszajn Theorem



A useful result that can be applied to RKHS is the Riesz Representation Theorem.



> **Theorem** The Riesz Representation Theorem
>
> If $L: \mathcal{H} \rightarrow \mathbb{R}$ is a bounded linear functional, then there is a unique vector $g \in \mathcal{H}$ such that a 
> $$
> \begin{eqnarray}
> L(f) = \big< f,g \big>,\forall f \in \mathcal{H}.
> \end{eqnarray}
> $$
> Moreover, $$\|L\| = \|g\|$$.



Since any evaluation functional $L_x$ of RKHS is *bounded linear functional*, we can find a function $R_x$ in RKHS such that 
$$
\begin{eqnarray}
f(x) = L_x(f) = \big< f, R_x\big>, \forall f \in \mathcal{H}.
\end{eqnarray}
$$

* Note1: $R_x$ also lives in RKHS. So if we have two points $x_i, x_j \in \mathcal{X}$
  $$
  \begin{eqnarray}
  R_{x_i}(x_j) &= L_{x_j}(R_{x_i})\\
  &= \big< R_{x_i}, R_{x_j} \big>\\
  &= \big< R_{x_j}, R_{x_i} \big>\\
  &= L_{x_i}(R_{x_j})\\
  &= R_{x_j}(x_i).
  \end{eqnarray}
  $$
  And this can be seen as an inner product of two maps (in fact these are **feature maps**) $R_{x_i}, R_{x_j}\in \mathcal{H}$. In equation (11), we saw that an inner product of two feature maps is a kernel.

* Note2: The kernel in equation (20) is called **reproducing kernel**.



## Moore-Aronszajn Theorem

Now our last direction is kernel-RKHS, i.e. a kernel defines RKHS uniquely.



> **Theorem** Moore-Aronszajn Theorem
>
> Given a kernel $k(\cdot, \cdot)$, there exists a  unique Hilbret space of functions on $\mathcal{X}$ for which $k(\cdot,\cdot)$ is a **reproducing kernel.**

> **proof**. We want to construct a RKHS $\mathcal{H}$ from the set of functions $\{k(x,\cdot): x\in \mathcal{X}\}$ .
>
> Let's define $\mathcal{H}_0$ as a finite linear combinations of the form
> $$
> \begin{eqnarray}
> f(x) = \sum_{i=1}^n \alpha_i k(x_i, x)
> \end{eqnarray}
> $$
> where $n \in \mathbb{N}$, $\alpha_1, \cdots, \alpha_n \in \mathbb{R}$ and $x_1, \cdots, x_n \in \mathcal{X}$. Then $\mathcal{H}_0$ is a vector space. 
>
> Consider $\mathcal{H}$ which is the completion of $\mathcal{H}_0$. Then $\mathcal{H}$ consists of functions of the form
> $$
> \begin{eqnarray}
> f(x) = \sum_{i=1}^{\infty} \alpha_i k(x_i, x)
> \end{eqnarray}
> $$
> where $$\sum_{i=1}^{\infty}\alpha_i^2 k(x_i, x_i) < \infty$$ by Cauchy-Schwartz inequality.
>
>
>
> Now we need  **inner product** and **bounded for evaluation functional** for RKHS.
>
> First, **inner product** can be defined as
> $$
> \begin{eqnarray}
> \big< f, g \big> \dot{=} \sum_{i=1}^{\infty} \sum_{j=1}^{\infty} \alpha_i \beta_j k(x_i, x_j')
> \end{eqnarray}
> $$
> for $$f(x) = \sum_{i=1}^{\infty} \alpha_i k(x_i, x)$$ and $$g(x) = \sum_{j=1}^{\infty} \beta_j k(x_j',x)$$. Then, the symmetry and linearity of inner product follows from the definition of the inner product.
>
> For positive definiteness, 
>
> 1. For any $f\in \mathcal{H}$, $\big< f, f \big> = \alpha^{\top} K \alpha \ge 0$ 
>
> 2. If $\big< f , f \big> = \alpha^{\top} K \alpha = 0$, we check positive definiteness by observing a more component $y$. 
>    Since k is a kernel, 
>    $$
>    \begin{eqnarray}
>    \begin{bmatrix}
>        K       & c^{\top} \\
>        c       & k(y, y) 
>    \end{bmatrix} \succeq 0
>    \end{eqnarray}
>    $$
>    where $c = [k(x_1, y), \cdots, k(x_m,y)]^{\top}$.
>
>    Therefore, we have
>    $$
>    \begin{eqnarray}
>    \alpha^{\top}K\alpha + 2(c^{\top}\alpha)b + k(y,y)b^2 \ge 0\\
>    2(c^{\top}\alpha)b + k(y,y)b^2 \ge 0
>    \end{eqnarray}
>    $$
>    for all $b$.
>    If $c^{\top}\alpha >0$, then negative $b$ whose absolute value is sufficiently small makes equation (27) contradiction.
>
>    If $c^{\top}\alpha <0$, then positive $b$ whose absolute value is sufficiently small makes equation (27) contradiction.
>
>    Therefore, $f_m(y) =\sum_{i=1}^{m} \alpha_i k(x_i,y) = c^{\top}\alpha = 0, \forall m \in \mathbb{N}$  and $f(y) = \lim_{m\rightarrow \infty}f_m(y) = 0$  .
>
> To check **bounded for evaluation functionals**, 
> $$
> \begin{eqnarray}
> L_x(f) &= f(x)\\
> &= \sum_{i=1}^{m} \alpha_i k(x_i, x)\\
> &= \big< f, k(x, \cdot)\big>\\
> &\le \|f\|_{\mathcal{H}}\|k(x,\cdot)\|_{\mathcal{H}}\\
> &=  \|f\|_{\mathcal{H}} k(x,x).
> \end{eqnarray}
> $$
> To prove **uniqueness**, let $\mathcal{G}$ be another Hilbert space of functions for which $k(\cdot, \cdot)$ is a reproducing kernel. For any $x, y \in \mathcal{X}$,
> $$
> \begin{eqnarray}
> \big< k(x, \cdot), k(y, \cdot) \big>_{\mathcal{H}} = k(x,y) = \big< k(x, \cdot), k(y, \cdot) \big>_{\mathcal{G}}.
> \end{eqnarray}
> $$
> By linearity, $$\big< \cdot, \cdot \big>_{\mathcal{H}} = \big< \cdot, \cdot \big>_{\mathcal{G}}​$$ on the span of $$\{k(x,\cdot): x \in \mathcal{X}\}​$$. Therefore, $$\mathcal{H} \subseteq \mathcal{G}​$$ because $$\mathcal{G}​$$ is complete and contains $$\mathcal{H}_0​$$.
>
>
> Let $f \in \mathcal{G}$. Since $\mathcal{H}$ is a closed subspace of $\mathcal{G}$, we can write $f = f_{\mathcal{H}} + f_{\mathcal{H}^{\perp}}$ where $f_{\mathcal{H}} \in \mathcal{H}$ and $f_{\mathcal{H}^{\perp}}\in \mathcal{H}^{\perp}$. Now if $x \in \mathcal{X}$, since $k(\cdot, \cdot)$ is a reproducing kernel of $\mathcal{G}$, we have
> $$
> \begin{eqnarray}
> f(x) = \big< f, k(x,\cdot)\big> = \big< f_{\mathcal{H}}, k(x, \cdot) \big> = f_{\mathcal{H}}(x)
> \end{eqnarray}
> $$
> which shows that $f_{\mathcal{H}^{\perp}} = 0$ and this makes $\mathcal{G} \subseteq \mathcal{H}$. $\square$



