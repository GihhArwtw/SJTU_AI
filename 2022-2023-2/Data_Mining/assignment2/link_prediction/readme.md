# Tasks

In this project, you need to implement **Node2Vec** algorithm on given graph data to finish the link prediction task. We will provide a network with 16863 nodes and 46116 edges. You can divide the training set and verification set to learn an encoder, so that the encoder can predict the probability of connecting edges between each pair of nodes. The trained encoder will be tested with a test set consisting of node pairs obtained from the original network. You need to give the probability of each given node pair in the test set connecting in the original network (4 digits after the decimal point).

 The edges are stored in file ''./data/lab2_edges.csv'', and the incomplete implementation is stored in ''./src/main.py''. 

## Attention

 👉 **Remember that your score depends on the implementation correctness and the ranges that your AUC falls into.**👈 

For fairness, we set a few rules for you to obey --- violation against the rules will lead to scoring deduction:

1. Please do not use any existing clustering API directly from libraries(For example, Embedding and Clustering). Algorithms should be implemented on your own.
2. Please do not copy someone else's code. We will run a plagiarism check after submission. 
3. Do not use other datasets to train you encoder.
4. Do not use any pretrained model.
5. For us to reproduce your code, please follow the "Environment" instruction.
6. If GPU resources are needed, please refer to the announcement on Canvas: [主题: 使用计算资源的注意事项 (sjtu.edu.cn)](https://oc.sjtu.edu.cn/courses/53650/discussion_topics/123560)

## Environment

You need to use Python 3.x and numpy to write the model, and we highly recommend you to use "Python3.8.0"  "networkx 3.0"  "numpy 1.23.5" for us to reproduce your code.

## Dataset

There are 2 CSV files:

1. lab2_edges.csv:

   Each row represents a source node and a target node

2. lab2_test.csv:

   Test set containing 10246 node pairs. You can embed the nodes first to get low-dimensional features and train the encoder using these features.
   
3. lab2_truth.csv:

   300 truth labels provided for you to test your algorithm. 

   


## Submission

Your codes need to generate a prediction result in a CSV file format. The CSV file should be named as 'predictions.csv', and contain two columns: the first column name is "id," and the second is "probability". Each node pair in the test set owns its corresponding id, your prediction result should follow this id order. For example, for node pair (0, 12429, 655), if your prediction is 0.8000, then the corresponding row in the uploaded file should be 0, 0.8000.

Do not change the structure of the whole project folder. 

You can add any necessary descriptions in appendix.

The project folder should contain:

│  readme.md
│ 
├─data
│	  lab2_edges.csv
│      lab2_test.csv
│      lab2_truth.csv
│	  predictions.csv
└─src
        main.py



## Definition of AUC

Area under curve: 
$$
AUC(f)= \frac{\sum_{t_0 \in D^0}\sum_{t_1 \in D^1} \boldsymbol{1}[f(t_0)<f(t_1)]}{|D^0|\cdot |D^1|}
$$



## Appendix