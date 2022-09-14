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

# [Solution of Homework 4] Martingale and Stopping Time

## Doob's martingale inequality

Let $\set{X_t}_{t\ge 0}$ be a martingale with respect to itself where $X_t\ge 0$ for every $t$. Prove that for every $n\in\^N$, 
$$
\Pr{\max_{0\le t\le n} X_t\ge \alpha} \le \frac{\E{X_0}}{\alpha}.
$$
:::spoiler Hint
Consider the stopping time $\tau=\arg\min_{t\le n}\set{X_t\ge \alpha}$ or $\tau=n$ if $X_t<\alpha$ for all $0\le t\le n$.
:::
*Proof.* We define a stopping time $\tau$ when the first element that is greater that $\alpha$ occurs, otherwise set $\tau=n$. Formally,
$$ \tau=\left\{
\begin{matrix}
n & \max_{0\leq t\leq n} X_t< \alpha \\
\arg\min_{t\le n}\set{X_t\ge \alpha} & o.w.
\end{matrix}
\right.
$$
By definition of $\tau$, we have
$$
\Pr{\max_{0\le t\le n} X_t\ge \alpha} = \Pr{X_{\tau}\ge \alpha}
$$
$\tau$ is bounded, so we apply Optional Stopping Theorem to obtain that $E[X_{\tau}]=E[X_0]$. Therefore, by Markov's Inequality,
$$
\Pr{\max_{0\le t\le n} X_t\ge \alpha}= \Pr{X_{\tau}\ge \alpha}\le \frac{\E{X_{\tau}}}{\alpha}= \frac{\E{X_0}}{\alpha}
$$

## Biased one-dimensional random walk

We study the biased random walk in this exercise. Let $X_t=\sum_{i=1}^tZ_i$ where each $Z_i\in\set{-1,1}$ is independent, and satisfies $\Pr{Z_i=-1}=p\in (0,1)$. 
* Define $S_t=\sum_{i=1}^t(Z_i+2p-1)$. Show that $\set{S_t}_{t\ge 0}$ is a martingale.
*Proof.* 
\begin{align*}
&\phantom{=}\E{S_t\mid Z_1,Z_2,\dots,Z_{t-1}}\\
&=\E{Z_t\mid Z_1,Z_2,\dots,Z_{t-1}} +  (2p-1)+S_{t-1}\\
&=\E{Z_t} +  (2p-1)+S_{t-1}\\
&=1-p-p +  (2p-1)+S_{t-1}\\
&=S_{t-1}.
\end{align*}
So $\set{S_t}_{t\ge 0}$ is a martingale.
* Define $P_t=\tp{\frac{p}{1-p}}^{X_t}$. Show that $\set{P_t}_{t\ge 0}$ is a martingale.
*Proof.*
\begin{align*}
&\phantom{=}\E{P_t\mid Z_1,Z_2,\dots, Z_{t-1}}\\
&=P_{t-1}\E{\left(\frac{p}{1-p}\right)^{Z_t}\mid Z_1, Z_2,\dots, Z_{t-1}}\\
&=P_{t-1}\E{\left(\frac{p}{1-p}\right)^{Z_t}}\\
&=P_{t-1}\left(\frac{p}{1-p}(1-p)+\frac{1-p}{p}p\right)\\
&=P_{t-1}
\end{align*}
* Suppose the walk stops either when $X_t=-a$ or $X_t=b$ for some $a,b>0$. Let $\tau$ be the stopping time. Compute $\E{\tau}$.
*Proof.*
When $p=\frac{1}{2}$, we've showed that $\E{\tau}=ab$, so we suppose $p\neq \frac{1}{2}$ in the following proof.
Consider a time period of length $T=a+b$. In each period of time, the walk stops with probability at least $p^{a+b}+(1-p)^{a+b}$. If we divide the time into consecutive periods in this manner, in expected finite time, we can meet some period with the event happened. Therefore, $\E{\tau}<\infty$. And 
\begin{align*}
|P_t-P_{t-1}|&=\left(\frac{p}{1-p}\right)^{X_{t}}+\left(\frac{p}{1-p}\right)^{X_{t-1}}\\
&<2\max\left(\left(\frac{p}{1-p}\right)^{-a},\left(\frac{p}{1-p}\right)^{b}\right),
\end{align*}
saying that $|P_t-P_{t-1}|$ is bounded by constant. So we apply OST and obtain that 
$$
\Pr{X_{\tau}=-a}\left(\frac{p}{1-p}\right)^{-a}+\Pr{X_{\tau}=b}\left(\frac{p}{1-p}\right)^{b}=\E{P_{\tau}}=\E{P_0}=1.
$$ 
Solving this equation, we get  $\Pr{X_{\tau}=-a}=\frac{1-\left(\frac{p}{1-p}\right)^{b}}{\left(\frac{p}{1-p}\right)^{-a}-\left(\frac{p}{1-p}\right)^{b}}$. Since $|S_t-S_{t-1}|=|Z_t+2p-1|<2$, applying OST, it follows that
$$
\Pr{X_{\tau}=-a}(-a)+\Pr{X_{\tau}=b}b+\E{\tau}(2p-1)=\E{S_{\tau}}=\E{S_0}=0.
$$
So when $p\neq \frac{1}{2}$,
$$
\E{\tau}=\frac{1-\left(\frac{p}{1-p}\right)^{b}}{\left(\frac{p}{1-p}\right)^{-a}-\left(\frac{p}{1-p}\right)^{b}}\frac{a+b}{2p-1}-\frac{b}{2p-1}.
$$

##  Longest common subsequence

A *subsequence* of a string $s$ is any string that can be obtained from $s$ by removing a few characters (not necessarily continuous). Consider two uniformly random strings $x,y\in\set{0,1}^n$. Let $X$ denote the length of their *longest common subsequence*.
* Show that there exist two constants $\frac{1}{2}<c_1<c_2<1$ such that $c_1 n <\E{X}<c_2 n$ for sufficiently $n$.
*Proof.*
Let $x=(x_1,x_2,\dots, x_n)$ and $y=(y_1,y_2,\dots,y_n)$. For the lower bound, assuming that the length of common subsequence of $(x_{2k-1},x_{2k})$ and $(y_{2k-1},y_{2k})$ is $l_k$, we have
$$
\E{X}\ge\E{\sum_{k=1}^{\lfloor\frac{n}{2}\rfloor} l_k}=\left(\frac{1}{2}\left(\frac{1}{2}+\frac{1}{4}2\right)+\frac{1}{2}\left(\frac{3}{4}+\frac{1}{4}2\right)\right)\left\lfloor\frac{n}{2}\right\rfloor=\frac{9}{8}\left\lfloor\frac{n}{2}\right\rfloor.
$$
So we could take $c_1=\frac{9}{16}$.
For the upper bound,
$$
\E{X}\leq \Pr{X\geq c_2 n}n+\Pr{x<c_2n}c_2n=(\Pr{X\geq c_2n}(1-c_2)+c_2)n.
$$
And by Stirling's approximation,
\begin{align*}
\Pr{X\geq c_2n} \leq \frac{\binom{n}{c_2n}^2}{2^{c_2n}} \sim \frac{1}{2\pi (1-c_2)c_2n}\left(\frac{1}{(\sqrt{2}c_2)^{c_2}(1-c_2)^{1-c_2}}\right)^{2n},
\end{align*}
for suffiently large $n$. We take $c_2=0.91$ so that $\Pr{X\geq c_2n}$ is $o(1)$ and then $\E{X}\le c_2n$ when $n$ is sufficiently large.

* Prove that $X$ is concentrated around $\E{X}$.
*Proof.*
We could regard $X$ as a function of $x_1,x_2,\dots,x_n,y_1,y_2,\dots,y_n$.
$$X=f(x_1,x_2,\dots,x_n,y_1,y_2,\dots,y_n).$$
And obviously $f$ is $1$-Lipschitz function since changing exactly one character in $x,y$ only add or delete at most one character in the longest common subsequences. Therefore, by Mcdarmid's Inequality,
$$\Pr{X-\E{X}\geq t}\leq 2e^{\frac{-t^2}{n}},$$
which means that $X$ is concentrated around $\E{X}$.




