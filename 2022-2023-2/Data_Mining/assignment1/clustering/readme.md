# Tasks

In this project, you are given 50000 papers’ feature vectors and each vector has 100 dimensions. These papers belong to 5 areas, e.g., Data Mining, Knowledge Management, Operation Research, Information Retrieval, Natural Language Processing. We have already hidden the ground truth of the areas, and your task is to complete the given implementation of a k-means clustering algorithm in Python to divide these papers into 5 clusters. The feature vectors are stored in file ''./data/features.csv'', and the incomplete implementation is stored in ''./src/main.py''. 

## Attention

 👉 **Remember that your score depends on the implementation correctness, the ranges that your accuracy falls into, and the running time of your algorithm.**👈 

For fairness, we set a few rules for you to obey --- violation against the rules will lead to scoring deduction:

1. Please do not use any existing clustering API directly from libraries. Algorithms should be implemented on your own.
2. Please do not copy someone else's code. We will run a plagiarism check after submission.
3. Please do not download the dataset somewhere else to train your model. A violation would be considered if there is a big gap between your reported results and our reproduced results.
4. Please do not use the pre-trained model. Everything should be built from scratch.
5. For us to reproduce your code, please follow the "Environment" instruction.

## Environment

You need to use Python 3.x and numpy to write the model, and we highly recommend you to use "Python3.8.0" and "numpy 1.23.5" for us to reproduce your code.

## Dataset

There is one CSV file containing all papers’ feature vectors. 

First, you need to divide all papers into 5 categories using a clustering algorithm. 

Second, you need to sort the clusters according to their radiuses (Euclidean distance\) in ascending order. 

Last, you should label each paper according to the cluster indices. For example, if the ordered cluster list is [[paper1, paper6], [paper2, paper8], [paper3, paper7], [paper4, paper9], [paper5, paper10]], you should provide its cluster indices as follows: {paper1: 0, paper2: 1, paper3: 2, paper4: 3, paper5: 4, paper6: 0, paper7: 2, paper8: 1, paper9: 3, paper10: 4}.

## Submission

Your codes need to generate a prediction result in a CSV file format. The CSV file should be named as 'predictions.csv', and contain two columns: the first column name is "ID," and the second is "Category."

Do not change the structure of the whole project folder. 

You can change the content of this readme file to append any necessary description.

The project folder should contain:

│  readme.md
│ 
├─data
│      features.csv
│      predictions.csv
│
└─src
        main.py



## Definition of Accuracy

Category Accuracy: $CA = \frac{N_{true}}{N}$ 

, where N is the number of papers, and N_true is the number of papers whose category index is equal to the ground truth. 

 Typically, clustering algorithms do not have `accuracy.’ Here we just need a way to evaluate your algorithm performance.