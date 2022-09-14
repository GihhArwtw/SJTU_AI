---
tags: AI2613, homework, 2022Fall
---

$$
\def\*#1{\mathbf{#1}} \def\+#1{\mathcal{#1}} 
\def\-#1{\mathrm{#1}}\def\^#1{\mathbb{#1}}\def\!#1{\mathtt{#1}}
\newcommand{\norm}[1]{\left\Vert#1\right\Vert}
\newcommand{\abs}[1]{\left\vert#1\right\vert}
\newcommand{\set}[1]{\left\{#1\right\}}
\newcommand{\tuple}[1]{\left(#1\right)} \newcommand{\eps}{\varepsilon}
\newcommand{\inner}[2]{\langle #1,#2\rangle} \newcommand{\tp}{\tuple}
\renewcommand{\mid}{\;\middle\vert\;} \newcommand{\cmid}{\,:\,}
\newcommand{\numP}{\#\mathbf{P}} \renewcommand{\P}{\mathbf{P}}
\newcommand{\defeq}{\triangleq} \renewcommand{\d}{\,\-d}
\newcommand{\ol}{\overline}
\newcommand{\Pr}[2][]{\mathbf{Pr}_{#1}\left[#2\right]}
\newcommand{\E}[2][]{\mathbf{E}_{#1}\left[#2\right]}
\newcommand{\Var}[2][]{\mathbf{Var}_{#1}\left[#2\right]}
\renewcommand{\emptyset}{\varnothing}
$$

# [Homework 4] Martingale (Due: May 8, 2022)

## Problem 1 (Doob's martingale inequality)

Let $\set{X_t}_{t\ge 0}$ be a martingale with respect to itself where $X_t\ge 0$ for every $t$. Prove that for every $n\in\^N$, 
$$
\Pr{\max_{0\le t\le n} X_t\ge \alpha} \le \frac{\E{X_0}}{\alpha}.
$$
:::spoiler Hint
Consider the stopping time $\tau=\arg\min_{t\le n}\set{X_t\ge \alpha}$ or $\tau=n$ if $X_t<\alpha$ for all $0\le t\le n$.
:::

## Problem 2 (Biased one-dimensional random walk)

We study the biased random walk in this exercise. Let $X_t=\sum_{i=1}^tZ_i$ where each $Z_i\in\set{-1,1}$ is independent, and satisfies $\Pr{Z_i=-1}=p\in(0,1)$. 
* Define $S_t=\sum_{i=1}^t(Z_i+2p-1)$. Show that $\set{S_t}_{t\ge 0}$ is a martingale.
* Define $P_t=\tp{\frac{p}{1-p}}^{X_t}$. Show that $\set{P_t}_{t\ge 0}$ is a martingale.
* Suppose the walk stops either when $X_t=-a$ or $X_t=b$ for some $a,b>0$. Let $\tau$ be the stopping time. Compute $\E{\tau}$.

##  Problem 3 (Longest common subsequence)

A *subsequence* of a string $s$ is any string that can be obtained from $s$ by removing a few characters (not necessarily continuous). Consider two uniformly random strings $x,y\in\set{0,1}^n$. Let $X$ denote the length of their *longest common subsequence*.
* Show that there exist two constants $\frac{1}{2}<c_1<c_2<1$ such that $c_1 n <\E{X}<c_2 n$ for sufficiently $n$.
* Prove that $X$ is well-concentrated around $\E{X}$ using tools developed in the class.

:::spoiler Hint
To find $c_2$, you can try to estimate the following probabilty: there exist $S,T\subset [n]$ such that (1) $\abs{S}=\abs{T}$ and both two sets are *large*; and (2) $x_S = y_T$ where $x_S$ and $y_T$ are the restrictions of $x$ and $y$ on $S$ and $T$ respectively.
:::

<br></br>
>【选做题，联动AI2615】设计一个$O(n^2)$的动态规划算法计算$x$与$y$的最长公共子序列
