import csv
import numpy as np
from numpy import argmax
from numpy.linalg import norm


DIMENSION = 100
TOTAL = 50000
CATEGORY = 5
THRESHOLD = 100

RANDOM_INIT_MEANS = False
# random initialing is faster but may fall into error
# while dispersed initialing costs time but has a stable performance.

data = np.zeros((TOTAL, DIMENSION))
means = np.zeros((CATEGORY, DIMENSION))
label = np.zeros(TOTAL, dtype=int)

def getData():
    global data
    with open("../data/features.csv", 'r') as csvFile:
        csvFile.readline()
        csv_reader = csv.reader(csvFile)
        PID = 0
        for row in csv_reader:
            data[PID] = row[1:] 
            PID += 1

def initRandomMeans():
    global means
    choice = np.random.choice(TOTAL, CATEGORY, replace=False)
    means = data[choice]

def initDispersedMeans():
    global means
    choice = np.zeros(CATEGORY, dtype=int)
    choice[0] = np.random.randint(TOTAL)
    distance2Means = np.full(50000, np.infty)
    for i in range(CATEGORY-1):
        for candidate in range(TOTAL):
            distance2Means[candidate] = min(distance2Means[candidate], norm(data[candidate] - data[choice[i]]))
        choice[i+1] = argmax(distance2Means)
    means = data[choice]

def initMeans():
    if RANDOM_INIT_MEANS:
        initRandomMeans()
    else:
        initDispersedMeans()

def storeResult(filename, labelMap):
    with open(filename, 'w') as output:
        output.write("id,category\n")
        for i in range(TOTAL):
            output.write("{},{}\n".format(i, labelMap[label[i]]))

def mapLabelName():
    ### return the map: current cluster ID -> radius-sorted cluster ID
    radius = np.zeros(CATEGORY)
    ### TODO: calculate radius of each cluster and store them in var:radius ###

    global data, means, label
    dist = norm(data - means[label], axis=1)
    for i in range(CATEGORY):
        radius[i] = np.max(dist[label == i])

    ### end of TODO ###
    temp =  radius.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(CATEGORY)
    print("Radius: ")
    print(radius[temp])
    return ranks

### TODO ###
### you can define some useful function here if you want

expand_data = np.expand_dims(data, axis=1)

def updateMetric():
    # update the metric of each cluster with Expectation-Maxmization
    global expand_data, means, label
    expand_means = np.expand_dims(means, axis=0).repeat(TOTAL, axis=0)
    dist = norm(expand_data - expand_means, axis=-1)
    label = np.argmin(dist, axis=-1)
    for i in range(CATEGORY):
        means[i] = np.mean(data[label == i], axis=0)

### end of TODO ###
        

def main():
    getData()
    initMeans()
    ### TODO ###
    # implement your clustering alg. here

    for _ in range(THRESHOLD):
        label_bkup = label
        updateMetric()
        if np.array_equal(label, label_bkup):
            break

    ### end of TODO ###
    labelMap = mapLabelName()
    storeResult("../data/predictions.csv", labelMap)


if __name__ == "__main__":
    main()
        