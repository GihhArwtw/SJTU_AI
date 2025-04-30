# Tasks
This project demonstrates using the MinHash algorithm to search a large collection of documents to identify pairs of documents which have a lot of text in common. You need to complete the implementation of MinHash. The incomplete implementation is stored in  main.py. The specific requirements are provided in main.py with a `TODO` mark.

**Note**: You cannot change any code outside the `TODO` part. A few codes inside the `TODO` part are just hints, and you can change them as you want.

## Attention

Please find the Attention section in ../clustering/readme.md

## Environment

You need to use Python 3.x to write the model, and we highly recommend you to use "Python3.8.0" for us to reproduce your code.

## Dataset
The format of articles_1000.train:
It is a plain text file. Each line contains an article. The first word of a line is the article ID (e.g. t3820), and the rest part is the content of the article.

## Submission

Do not change the structure of the whole project folder. 

You can change the content of this readme file to append any necessary description.

The project folder should contain:
│  readme.md
│
├─data
│      articles_1000.train
│      articles_1000.truth
│      prediction.csv
│
└─src
        main.py