# HW: Differentially Private Stochastic Gradient Descent

Teaching Assistant: yangjungang@sjtu.edu.cn

## Concept Questions

Please provide your solutions to the following questions in a single PDF file. 

1. Calculate $b$ from the Laplace distribution $Lap(x|b)$ that satisfies ε-differential privacy with an l1-sensitivity of 1.

2. Describe the algorithm for Differentially Private Stochastic Gradient Descent.

   Hint: Refer to Algorithm 1 in "Deep Learning with Differential Privacy."

3. Design an algorithm to enhance differentially private SGD based on the following requirements. Let $g(x_i) \in R^{p}$ be the gradient of the example $x_i$, and $G \in R^{n \times p} = [g(x_1) g(x_2) ... g(x_n)]$ be the gradient matrix. Create an algorithm to compress the gradient matrix such that $\hat{G} = GB$, where $\hat{G} \in R^{n \times k}, k < p$, and $B\in R^{P \times k}$ is a direction matrix, related to the direction of $G$, and it needs to be guaranteed to be orthogonal. Utilize $\hat{G}$  to perform per-example clipping and add Gaussian noise in DPSGD. Finally, project the noise gradient back to $R^p$ using $B^T$ and update the model's parameters. Provide the algorithm for the entire process.

   Hint: For gradient compression methods, you could refer to SVD.

## Coding Problems

In the `src` directory, the `main.py` file provides code for DPSGD to train a CNN on the MNIST dataset using PyTorch. Some parts of the code related to differential privacy are missing. Please complete these sections to ensure proper functionality.

1. Complete the code for Differentially Private Stochastic Gradient Descent.

   a. Fill in the code for per-example clipping and adding Gaussian noise.

   b. Implement the privacy budget composition. Calculate the privacy budget of the training process, which means calculating $\epsilon$ based on the variance of Gaussian noise $\sigma^2$ and the given $\delta = 10^{-5}$ in different epochs. You can use basic composition to complete the code. If you correctly apply the Moments Accountant method, you will receive bonus points.
   
2. Gradient compression during the training process.
   Following the algorithm from Concept Question 4, we need to enhance DPSGD with gradient compression. Complete the function to compress the gradient from $R^p$ to $R^k$. Then, calculate the average gradient and add Gaussian noise $Z \in R^k$ to the compressed average gradient. And complete the training process.

The code can be executed with the default hyperparameter settings:

`python main.py`

To run without DPSGD:

`python main.py --disable-dp`

To run DPSGD with gradient compression:

`python main.py --using-compress`

You can also adjust other hyperparameters such as learning rate, batch size, epochs, clip value, gradient project dimension, and power_iter (optional) to achieve higher accuracy and lower privacy budget. To ensure the fairness of the experiment, you CANNOT change the hyperparameter setting for sigma=1.
The recommended range for gradient project dimension is between 16 and 2048.

We will execute your code using `python main.py` and `python main.py --using-compress`. Please ensure that you set the best hyperparameters as the default when you submit your code.

## Submission

Please submit a zip file named `[name]_[studentID]_hw4.zip`. It should contain one folder named `[name]_[studentID]_hw4`. 

This folder should include one PDF file for the concept questions and one `src` folder for the coding problems. 

In the `src` folder, please only include the '*.py' files. DO NOT upload the datasets or training results!

## Scoring Criteria
The scoring criteria for the concept questions are as follows: 10 points for the first question, 10 points for questions 2, and 20 points for question 3.

| Question | Scoring Criteria |
|----------|-----------------|
|   1      |      10 points  |
|   2      |      10 points  |
|   3      |      20 points  |

The scoring criteria for the coding problems are as follows: 15 points for 1.a, 10 points for 1.b, 15 points for 2, and 20 points for the final results.

| Coding Problems | Scoring Criteria |
|----------|-----------------|
|   1.a      |       15 points  |
|   1.b      |      10 points  |
|   2      |      15 points  |
|   Results for 1     |      10 points  |
|   Results for 2     |      10 points  |

For the experimental results you need to consider all three aspects: differential privacy, accuracy and runtime. The DPSGD training should be conducted with $\epsilon < 50$, and $\epsilon < 25$ is better. Additionally, the accuracy scoring criteria will be divided according to accuracy ranges, for example Acc $\ge 95\%$, Acc $\ge 90\%$, etc.

## Supplementary Materials

We have provided a file named `piplist.txt`, which contains the environment you may need.
