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

# [Homework 2]: Finite Markov Chains, Coupling

## Problem 1 (Optimal Coupling)

Let $\Omega$ be a finite state space and $\mu,\nu$ be two distributions over $\Omega$. Prove that there exists a coupling $\omega$ of $\mu$ and $\nu$ such that
$$
\Pr[(X,Y)\sim \omega]{X\ne Y} = D_{\-{TV}}(\mu,\nu).
$$
You need to explicitly describe how $\omega$ is constructed.

## Problem 2 (Stochastic Dominance)

Let $\Omega\subseteq \^Z$ be a finite set of integers. Let $\mu$ and $\nu$ be two distributions over $\Omega$. We say  $\mu$ is *stochastic dominance* over $\nu$ if for $X\sim \mu$, $Y\sim\nu$ and any $a\in\Omega$, 
$$
    \Pr{X\ge a}\ge \Pr{Y\ge a}.
$$
We write $\mu\succeq\nu$.

* Consider the binomial distirbution $\!{Binom}(n,p)$ where $X\sim \!{Binom}(n,p)$ satisfies for any $a=0,1,\dots, n$, $\Pr{X=a} = \binom{n}{a}\cdot p^a\cdot (1-p)^{n-a}$. Prove that for any $p,q\in[0,1]$, $\-{Binom}(n,p)\succeq \-{Binom}(n,q)$ if and only if $p\ge q$.
* A coupling $\omega$ of $\mu$ and $\nu$ is *monotone* if $\Pr[(X,Y)\sim \omega]{X\ge Y}=1$. Prove that $\mu\succeq\nu$ if and only if a monotone coupling of $\mu$ and $\nu$ exists.
* Consider the [Erdős–Rényi](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model) model $\+G(n,p)$ for random graph. In this model, each $G\sim \+G(n,p)$ is a simple undirected random graph with $n$ vertices where each $\set{i,j}\in\binom{[n]}{2}$ is present with probability $p$ independently. Prove that for any $p,q\in[0,1]$ satisfying $p\ge q$, it holds that $\Pr[G\sim\+G(n,p)]{G\mbox{ is connected}} \ge \Pr[H\sim\+G(n,q)]{H\mbox{ is connected}}$.

## Problem 3 （Total Variation Distance is Non-Increasing）

Let $P$ be the transition matrix of an irreducible and aperiodic Markov chain with state space $\Omega$. Let $\pi$ be its stationary distribution. Let $\mu_0$ be an arbitrary distribution on $\Omega$ and $\mu_t^{\!T} = \mu_0^{\!T}P^t$ for every $t\ge 0$. For every $t\ge 0$, let $\Delta(t) = D_{\!{TV}}(\mu_t,\pi)$ be the total variation distance between $\mu_t$ and $\pi$. Prove that $\Delta(t+1)\le \Delta(t)$ for every $t\ge 0$.

