# Tasks

In this project, you need to implement Community Detection algorithm on given graph data. We will provide a directed citation network with 31136 nodes and 160000 edges, where the direction of the directed edge represents the citation relationship of the paper. These papers are divided into five categories (AAAI, IJCAI, CVPR, ICCV, ICML) according to their conference names, which are represented by 0-4 respectively. Your task is to implement the Louvain algorithm **and** PPR based algorithm to divide these papers into five categories.

 The edges are stored in file ''./data/lab1_edges.csv'', and the incomplete implementation is stored in ''./src/louvain.py'' and ''./src/PPR.py''. 

## Attention

 👉 **Remember that your score depends on the implementation correctness and the ranges that your accuracy falls into**👈 

For fairness, we set a few rules for you to obey --- violation against the rules will lead to scoring deduction:

1. Please do not use any existing clustering API directly from libraries. Algorithms should be implemented on your own.
2. Please do not copy someone else's code. We will run a plagiarism check after submission. 
5. For us to reproduce your code, please follow the "Environment" instruction.
4. If GPU resources are needed, please refer to the announcement on Canvas: [主题: 使用计算资源的注意事项 (sjtu.edu.cn)](https://oc.sjtu.edu.cn/courses/53650/discussion_topics/123560)

## Environment

You need to use Python 3.x and numpy to write the model, and we highly recommend you to use "Python3.8.0" "networkx 3.0" "numpy 1.23.5" for us to reproduce your code.

## Dataset

There are 2 CSV files:

1. edges.csv:

   Each row represents a source node and a target node

2. ground_truth.csv:

   Some ground truth labels provided for you to test your algorithm. The number of these nodes accounts for about 1% of the total nodes. For each paper category, there are 60 nodes.


## Submission

Your codes need to generate a prediction result in a CSV file format. The CSV file should be named as 'predictions.csv', and contain two columns: the first column name is "id" and the second is "category".  You can choose the best result of the two algorithms, but you must implement both two algorithms.

Do not change the structure of the whole project folder. 

You can add any necessary descriptions in appendix.

The project folder should contain:

│  readme.md
│ 
├─data
│      lab1_edges.csv
│      lab1_truth.csv
│	  predictions.csv
└─src

​		PPR.py

​        louvain.py



## Definition of Accuracy

Category Accuracy: $CA = \frac{N_{true}}{N}$ 

where N is the number of papers, and N_true is the number of papers whose category index is equal to the ground truth. 



## Appendix