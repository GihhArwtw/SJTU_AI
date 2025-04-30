import math


def ACC(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1
    return sum / float(len(real))


def MRR(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))


def NDCG(real, predict):
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i + 1
            dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    return dcg / float(idcg)


def IDCG(n):
    idcg = 0
    itemRelevance = 1
    for i in range(n): idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
    return idcg
