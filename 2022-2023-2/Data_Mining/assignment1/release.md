# HW1 High-Dimensional Data 


Teaching assistant:  ji_zhe@sjtu.edu.cn

## Concept questions

You should write your solutions for the following 4 questions in a single pdf file.

1. Please prove Jaccard distance is a metric (or not). 

   Hint: you may need to prove the lemma below at first:

   For any set X and its subsets A, B, C, i.e. $A,B,C\subseteq X$, it holds that

   $|A\cap C|\cdot |B\cup C| + |A\cup C| \cdot |B\cap C| \leq |C| \cdot(|A| + |B|)$ 

2. Prove the average distance between a pair of points on a line of length L is L/3. 

3. Let $A = U\Sigma V^T$  and $B = U S V^T$ where $S = $ diagonal $r \times r$ matrix with $s_i=\sigma_i (i=1...k)$， and $s_i=0$ otherwise. Please prove $B$ is a best $k$-rank approximation to $A$ in terms of Frobenius norm error.

4. Suppose we have a universal set U of n elements, and we choose two subsets S and T at random, each with m of the n elements. What is the expected value of the Jaccard similarity of S and T ?  A sum expression of the expectation is acceptable if you can't simplify it.



## Coding problems

1. Clustering.
1. Finding similar items.

Please find the detail in the corresponding readme files.



## Submission

You should submit a zip file named `[name]_[studentID]_hw1.zip` . It should contain exactly one folder that is named `[name]_[studentID]_hw1` . 

This folder should contain one pdf file for concept questions and two project folders for coding problems.

│  concepts.pdf
│
├─clustering
│
└─similarItem