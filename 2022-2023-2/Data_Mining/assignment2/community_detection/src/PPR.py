# please use PPR algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result

import networkx as nx
import csv
import random
from queue import Queue
import matplotlib.pyplot as plt
# you can use basic operations in networkx
# you can also import other libraries if you need, but do not use any community detection APIs

NUM_NODES = 31136

# read edges.csv and construct the graph
def getGraph():
    G = nx.DiGraph()
    for i in range(NUM_NODES):
        G.add_node(i)
    with open("../data/lab1_edges.csv", 'r') as csvFile:
        csvFile.readline()
        csv_reader = csv.reader(csvFile)
        for row in csv_reader:
            source = int(row[0])
            target = int(row[1])
            G.add_edge(source, target)
    print("graph ready")
    return G

# save the predictions to csv file
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
def storeResult(G):
    with open('../data/predictions_PPR.csv', 'w') as output:
        output.write("id,category\n")
        for i in range(NUM_NODES):
            output.write("{},{}\n".format(i, G._node[i]['category']))

# approximate PPR using push operation
def approximatePPR(G, beta=0.8, epsilon=1e-12, seed=-1):

    """
    Compute approximate PPR using push operation.

    Set seed to -1 to use a random seed.
    """
    
    if (seed not in G.nodes):
        seed = random.randint(0, NUM_NODES - 1)

    print("Picking NODE {} as the seed.".format(seed))

    ppr = {node: 0 for node in G.nodes()}
    res = {node: 0 for node in G.nodes()}
    
    deg = nx.degree(G)
    res[seed] = 1.
    queue = Queue()
    queue.put(seed)

    while not queue.empty():
        node = queue.get()
        q = res[node] / deg[node]
        if q <= epsilon:
            continue
        
        ppr_backup = ppr.copy()
        res_backup = res.copy()

        ppr[node] = ppr_backup[node] + (1. - beta) * res_backup[node]
        res[node] = beta * res_backup[node] / 2.
        for v in G.neighbors(node):
            res[v] = res_backup[v] + beta * q / 2.
            queue.put(v)
    
    return ppr


# use PPR to compute conductance and label the nodes
def sweepCut(G, ppr, plot=False):
    
    ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)

    cut = 0
    volume = 0
    
    last = None
    first_min = False

    phi = dict()

    for i in range(NUM_NODES):
        node = ppr[i][0]
        volume += ( G.in_degree(node) + G.out_degree(node) )
        cut += ( G.in_degree(node) + G.out_degree(node) )

        out_deg = 0
        in_deg = 0
        for v in G.successors(node):
            if v in phi.keys():
                out_deg += 1
        for v in G.predecessors(node):
            if v in phi.keys():
                in_deg += 1
        
        cut -= (out_deg + in_deg) * 2
        phi[node] = cut / volume

        if last is None:
            last = phi[node]
            G._node[node].update({'category': 0})
            continue

        if (phi[node] >= last):
            if not first_min:
                print("The first local minimum is at the node with {}-th largest PPR score, i.e. NODE {}".format(i, ppr[i][0]))
            first_min = True

        G._node[node].update({'category': int(first_min)})

        last = phi[node]

    # # label the nodes
    # for node in G.nodes():
    #     if conductance[node] > 0:
    #         G._node[node].update({'category': 1})
    #     else:
    #         G._node[node].update({'category': 0})

    if plot:
        print("Making plot...")
        phis = [phi[ppr[i][0]] for i in range(NUM_NODES)]
        fig = plt.figure(figsize=(8,6))
        plt.plot(list(range(NUM_NODES)), phis)
        plt.xlabel('order in increasing approximated PPR score')
        plt.ylabel('conductance')
        plt.title('conductance of nodes')
        plt.show()


### TODO ###
### you can define some useful function here if you want


### end of TODO ###


def main():
    G = getGraph()

    ### TODO ###
    # implement your community detection alg. here

    random.seed(999)
    ppr = approximatePPR(G, beta=0.8, epsilon=1e-7)
    print("ppr ready.")

    sweepCut(G, ppr, plot=True)

    ### end of TODO ###

    storeResult(G)

if __name__ == "__main__":
    main()