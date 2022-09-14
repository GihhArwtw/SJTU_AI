---
tags: AI2613, homework solution, 2022Fall
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

# [Solution of Homework 5] Poisson Process & Poisson Approximation 


## Problem 1
Customers arrive according to a Poisson process of rate $\lambda$ per hour. Joe does not want to stay until the store closes at $T=10$ p.m., so he decides to close up when the first customer after time $T-s$ arrives. He wants to leave early but he does not want to lose any business so he is happy if he leaves before $T$ and no one arrives after. 
* What is the probability he achieves his goal? 
* What is the optimal value of $s$ and the corresponding success probability? (That is, the value $s$ maximizing the success probability)

### (a)
:::success
**Solution**
The goal of Joe is the event that there is only one customer arrived within time $T-s \sim T$ (the unit of variable $s$ is hour). According to the fact that customers arrive according to a Poisson process of rate $\lambda$ per hour, we have
\begin{align*}
    \Pr{\text{Joe achieves his goal}}= s\lambda e^{-s\lambda}.
\end{align*}
:::

### (b)
:::success
**Solution**
We can maximize $s\lambda e^{-s\lambda}$ when $s=\frac{1}{\lambda}$ where the maximum value is $\frac{1}{e}$.
:::

## Problem 2
* Assume $X\sim\mathtt{Poisson}(\lambda)$ for some integer $\lambda\ge 1$. Prove that for any $k=0,1,\dots,\lambda-1$, it holds that $\Pr{X=\lambda+k}\ge \Pr{X=\lambda-k-1}$. Use this to conclude that $\Pr{X\ge \lambda}\ge \frac{1}{2}$.
* Recall the setting of Corollary 4 in Lecture 10. Prove that if $\E{f(X_1,\dots,X_n)}$ is monotonically increasing in $m$, then
$$
    \E{f(X_1,\dots,X_n)} \le 2\cdot\E{f(Y_1,\dots,Y_n)}.
$$
* Recall the birthday problem in Lecture 2 and assume notations there. Now suppose we would like to estimate the probability of the event "*there exists four students who share the same birthday*". Assume there are 50 students in the class ($n=50$ and $m=365$). Use Poisson approximation to show that the probabilty is at most $1\%$.


### (a)
:::success
**Solution**
By definition, we have
\begin{align*}
    \frac{\Pr{X=\lambda+k}}{\Pr{X=\lambda-k-1}}&= \frac{e^{-\lambda}\frac{\lambda^{\lambda+k}}{(\lambda+k)!}}{e^{-\lambda}\frac{\lambda^{\lambda-k-1}}{(\lambda-k-1)!}}\\
    &=\frac{\lambda^{2k+1}}{(\lambda+k)(\lambda+k-1)\cdots(\lambda-k)}\\
    &=\sum_{i=1}^k \frac{\lambda^2}{(\lambda+i)(\lambda-i)}\geq 1,
\end{align*} which certifies
$$\Pr{X=\lambda+k}\geq \Pr{X=\lambda-k-1}.$$
Then,
\begin{align*}
\Pr{X\geq \lambda}&= \frac{\sum_{t=\lambda}^\infty \Pr{X= t}}{\sum_{t=0}^\infty \Pr{X= t}}\\
    &\geq \frac{\sum_{t=\lambda}^{2\lambda-1} \Pr{X= t}}{\sum_{t=0}^{2\lambda-1} \Pr{X= t}}\\
    &=\frac{\sum_{t=\lambda}^{2\lambda-1} \Pr{X= t}}{\sum_{t=0}^{\lambda-1} \Pr{X= t}+\sum_{t=\lambda}^{2\lambda-1} \Pr{X= t}}\geq \frac{1}{2}.\\
\end{align*} 
:::

### (b)
:::success
**Solution**
For convenience, we use $\*E_m$ to denote the expectation according to the distribution in the $m$-balls-into-$n$-bins model.
\begin{align*}
    \E{f(Y_1,Y_2,\dots,Y_n)}&=\sum_{k=0}^{\infty} \E{f(Y_1,Y_2,\dots,Y_n)\mid \sum_{i=1}^n Y_i=k} \Pr{\sum_{i=1}^n Y_i=k}\\
    &\geq \sum_{k=m}^{\infty} \E{f(Y_1,Y_2,\dots,Y_n)\mid \sum_{i=1}^n Y_i=k} \Pr{\sum_{i=1}^n Y_i=k}\\
    &= \sum_{k=m}^{\infty} \E[k]{f(X_1,X_2,\dots,X_n)} \Pr{\sum_{i=1}^n Y_i=k}\\
    &\geq \E[m]{f(X_1,X_2,\dots,X_n)} \sum_{k=m}^{\infty}\Pr{\sum_{i=1}^n Y_i=k}.
\end{align*}

Note that $\sum_{i=1}^n Y_i\sim \text{Possion}(m)$ and 
$$\sum_{k=m}^{\infty}\Pr{\sum_{i=1}^n Y_i=k}\geq \frac{1}{2}.$$
Therefore,
$$
    \E{f(X_1,\dots,X_n)} \le 2\cdot\E{f(Y_1,\dots,Y_n)}.
$$
:::


### (c)
:::success
**Solution**
The birthday problem with $n$ students is exactly the $n$-balls-into-$m$-bins model. For any $i\in[m]$, we define 
$$
    X_i:= \text{the number of students born in day } i.
$$
Let $Y_i\sim \text{Possion}(\frac{n}{m})$ and 
$$
    f(X_1,\dots,X_m):=\*1[\max{\set{X_1,\dots,X_m}}\geq 4]
$$ which is monotone with respect to $n$. Then we have
\begin{align*}
    \Pr{\max{\set{X_1,\dots,X_m}}\geq 4}&\leq 2\cdot \Pr{\max{\set{Y_1,\dots,Y_m}}\geq 4}\\
    &=2\cdot(1-\Pr{\max{\set{Y_1,\dots,Y_m}}< 4})\\
    &=2\cdot (1-\prod_{i=1}^m\Pr{Y_i<4})\\
    &\leq 0.01.
\end{align*}
:::



