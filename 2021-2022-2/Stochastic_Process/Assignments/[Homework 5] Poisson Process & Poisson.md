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

# [Homework 5] Poisson Process & Poisson Approximation (Due: May 25, 2022)


## Problem 1
Customers arrive according to a Poisson process of rate $\lambda$ per hour. Joe does not want to stay until the store closes at $T=10$ p.m., so he decides to close up when the first customer after time $T-s$ arrives. He wants to leave early but he does not want to lose any business so he is happy if he leaves before $T$ and no one arrives after. 
* What is the probability he achieves his goal? 
* What is the optimal value of $s$ and the corresponding success probability? (That is, the value $s$ maximizing the success probability)

## Problem 2
* Assume $X\sim\mathtt{Poisson}(\lambda)$ for some integer $\lambda\ge 1$. Prove that for any $k=0,1,\dots,\lambda-1$, it holds that $\Pr{X=\lambda+k}\ge \Pr{X=\lambda-k-1}$. Use this to conclude that $\Pr{X\ge \lambda}\ge \frac{1}{2}$.
* Recall the setting of Corollary 4 in Lecture 10. Prove that if $\E{f(X_1,\dots,X_n)}$ is monotonically increasing in $m$, then
$$
    \E{f(X_1,\dots,X_n)} \le 2\cdot\E{f(Y_1,\dots,Y_n)}.
$$
* Recall the birthday problem in Lecture 2 and assume notations there. Now suppose we would like to estimate the probability of the event "*there exists four students who share the same birthday*". Assume there are 50 students in the class ($n=50$ and $m=365$). Use Poisson approximation to show that the probabilty is at most $1\%$.

