---
tags: AI2613, solution, 2022Spring
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

# [Solution of Homework 2]

## Problem 1 (Optimal Coupling)



Let $\Omega$ be a finite state space and $\mu,\nu$ be two distributions over $\Omega$. Prove that there exists a coupling $\omega$ of $\mu$ and $\nu$ such that
$$
\Pr[(X,Y)\sim \omega]{X\ne Y} = D_{\-{TV}}(\mu,\nu).
$$
You need to explicitly describe how $\omega$ is constructed.
*Proof.* 
    Let $P(x,y)$ denote $\Pr[(X,Y)\sim \omega]{X=x,Y=y}$ and use $\rho$ to denote $D_{\-{TV}}(\mu,\nu)$ for shorthand. The coupling $\omega$ can be constructed as follows: First, we set $P(a,a)=\min\{\mu(a),\nu(a)\},\forall a\in\Omega$. Clearly, if $D_{\-{TV}}(\mu,\nu)=0$, the forementioned setting is indeed the optimal coupling. Otherwise, for any $a\in \Omega$, let
    \begin{align*}
        &R_X(a)=\mu(a)-P(a,a)\\
        &R_Y(a)=\nu(a)-P(a,a).
    \end{align*}
For any $a,b\in \Omega$, let $$ P(a,b)=\frac{R_X(a)R_Y(b)}{\rho}.$$
			It's clear that $\sum_aR_X(a)=\sum_bR_Y(b)=\rho$ by the fact that $D_{\-{TV}}(\mu,\nu)=\max_{A\in \Omega}|\mu(A)-\nu(A)|$. Moreover, $R_X(a)R_Y(a)=0, \forall a\in\Omega$.
Now we need check it's indeed a coupling. For a fixed $a\in \Omega$,
			\begin{align*}
				\sum_bP(a,b)&=P(a,a)+\sum_{b:b\neq a}P(a,b)\\
				&=P(a,a)+\sum_{b:b\neq a}\frac{R_X(a)R_Y(b)}{\rho}\\
				&=P(a,a)+\frac{R_X(a)}{\rho}(\rho-R_Y(a))\\
				&=P(a,a)+R_X(a)=\mu(a)
			\end{align*}
			Similarly, you can check that $\sum_bP(a,b)=\nu(b),\forall b\in\Omega$. Hence it's a feasible coupling. As for the optimiality,
    \begin{align*}
        \Pr[(X,Y)\sim \omega]{X\ne Y}&=1-\sum_{a\in \Omega} P(a,a)\\
        &= \sum_{a\in \Omega} \mu(a)- \sum_{a\in \Omega}P(a,a)=\rho.
    \end{align*}






## Problem 2 (Stochastic Dominance)

Let $\Omega\subseteq \^Z$ be a finite set of integers. Let $\mu$ and $\nu$ be two distributions over $\Omega$. We say  $\mu$ is *stochastic dominance* over $\nu$ if for $X\sim \mu$, $Y\sim\nu$ and any $a\in\Omega$, 
$$
    \Pr{X\ge a}\ge \Pr{Y\ge a}.
$$
We write $\mu\succeq\nu$.


    
1. Consider the binomial distirbution $\!{Binom}(n,p)$ where $X\sim \!{Binom}(n,p)$ satisfies for any $a=0,1,\dots, n$, $\Pr{X=a} = \binom{n}{a}\cdot p^a\cdot (1-p)^{n-a}$. Prove that for any $p,q\in[0,1]$, $\-{Binom}(n,p)\succeq \-{Binom}(n,q)$ if and only if $p\ge q$.
    *Proof.* 
    We consturct the following coupling with respect to $\-{Binom}(n,p)$ and  $\-{Binom}(n,q)$:
        1. Sample $U_i$ uniformly at random from $[0,1]$ for any $i\in[n]$ i.i.d;
        2. $X_i=1$ iff $U_i\le p$ and $Y_i=1$ iff $U_i\le q$ for any $i\in[n]$;
        3. Let $X=\sum_{i=1}^{n}X_i$ and $Y=\sum_{i=1}^{n}Y_i$.
    It is obvious that $X\sim\-{Binom}(n,p)$ and $Y\sim \-{Binom}(n,q)$ which justifies the above process is indeed a coupling. With this coupling, we know that $\set{Y\geq a}\subseteq \set{X\geq a}$ for any $a=0,1,\dots,n$ iff $p\ge q$. Therefore, if $p\ge q$, we have $\-{Binom}(n,p)\succeq \-{Binom}(n,q)$.
    
    
    On the other hand, let $X\sim \-{Binom}(n,p)$ and $Y\sim \-{Binom}(n,q)$. If $\-{Binom}(n,p)\succeq \-{Binom}(n,q)$, we have $\Pr{X=n}\geq \Pr{Y=n}$ which implies that $p\ge q$.
    
2. A coupling $\omega$ of $\mu$ and $\nu$ is *monotone* if $\Pr[(X,Y)\sim \omega]{X\ge Y}=1$. Prove that $\mu\succeq\nu$ if and only if a monotone coupling of $\mu$ and $\nu$ exists.
   *Proof.* 
   Proof of "$\Leftarrow$".Suppose $\omega$ is a monotone coupling of $\mu$ and $\nu$, which means $\Pr[(X,Y)\sim\omega]{X\geq Y}=1$. Then
		\begin{align*}
		\Pr[Y \sim \nu]{Y\geq a}&=\Pr[(X, Y) \sim \omega]{Y \geq a} \\
		&=\Pr[(X,Y) \sim \omega]{X \geq Y \wedge Y \geq a}+\Pr[(X, Y) \sim \omega]{X<Y \wedge Y \geq a} \\
		&=\Pr[(X, Y) \sim \omega]{X \geq Y \geq a} \\
		&\leq \Pr[(X, Y) \sim \omega]{X \geq a}=\Pr[X \sim \mu]{X \geq a}.
		\end{align*}
		
        
    Proof of "$\Rightarrow$". Define the cumulative distribution function $F_\mu(x)=\mu((-\infty,x])$ and $F_\nu(x)=\nu((-\infty,x])$. Then use these functions to construct two random variables:
		$$X=F_\mu^{-1}(U), Y=F_\nu^{-1}(U),$$
		where $U$ is sampled uniform at random from $[0,1]$ and $F_\mu^{-1}(u)\triangleq\inf \left\{x \in \mathbb{R}: F_{\mu}(x) \geq u\right\}$ is a generalized inverse(similar for $F_\nu^{-1}$).		
		Now we claim that $\omega=(X,Y)$ is a monotone coupling of $\mu$ and $\nu$.
		First we need to check that $\omega$ is a coupling of $\mu$ and $\nu$ which means $X$ and $Y$ follows $\mu$ and $\nu$ respectively.
		$$\Pr{X\leq x}=\Pr{F_\mu^{-1}(U)\leq x}=\Pr{U\leq F_\mu (x)}=F_\mu(x) \quad(\text{similar for Y})$$
		Then we will check $\+C$ is a monotone coupling. According to $\mu\succeq \nu$, we have $F_\mu(x)\leq F_\nu(x)$ for $x\in\^R$. Thus
		$$\Pr{X\geq Y}=\Pr{F_\mu^{-1}(U)\geq F_\nu^{-1}(U)}=1.$$
        
3. Consider the [Erdős–Rényi](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model) model $\+G(n,p)$ for random graph. In this model, each $G\sim \+G(n,p)$ is a simple undirected random graph with $n$ vertices where each $\set{i,j}\in\binom{[n]}{2}$ is present with probability $p$ independently. Prove that for any $p,q\in[0,1]$ satisfying $p\ge q$, it holds that $\Pr[G\sim\+G(n,p)]{G\mbox{ is connected}} \ge \Pr[H\sim\+G(n,q)]{H\mbox{ is connected}}$.
    *Proof*
    The following coupling justifies the statement:
        1. Sample $U_e$ uniformly at random from $[0,1]$ for any $e\in\binom{[n]}{2}$ i.i.d;
        2. $e$ occurs in $G$ iff $U_e\le p$ and $e$ occurs in $H$ iff $U_e\le q$ for any $e\in\binom{[n]}{2}$.
        


## Problem 3 （Total Variation Distance is Non-Increasing）

Let $P$ be the transition matrix of an irreducible and aperiodic Markov chain with state space $\Omega$. Let $\pi$ be its stationary distribution. Let $\mu_0$ be an arbitrary distribution on $\Omega$ and $\mu_t^{\!T} = \mu_0^{\!T}P^t$ for every $t\ge 0$. For every $t\ge 0$, let $\Delta(t) = D_{\!{TV}}(\mu_t,\pi)$ be the total variation distance between $\mu_t$ and $\pi$. Prove that $\Delta(t+1)\le \Delta(t)$ for every $t\ge 0$.
*Proof*
    Let $X_t\sim \mu_t$ and $Y_t\sim \pi$. By coupling lemma, there exist a coupling $\omega$ of $X_t$ and $Y_t$ such that $\Pr{X_t\neq Y_t}=\Delta(t)$. Equipped with $\omega$, we construct the coupling $\omega'$ of $X_{t+1}$ and $Y_{t+1}$ as follows:
    1. We first sample $(X_t,Y_t)$ from $\omega$;
    2. Next we run the Markov chain according to the transition matrix $P$ on $X_t$ and $Y_t$ as follows:
* If $X_t=Y_t$, the two chains evolve synchronously;
* If $X_t\neq Y_t$, the two chains evolve independently.

Under the coupling $\omega'$,
    \begin{align*}
        \Delta(t+1)\le \Pr{X_{t+1}\neq Y_{t+1}}\leq \Pr{X_{t}\neq Y_{t}}=\Delta(t).
    \end{align*}
