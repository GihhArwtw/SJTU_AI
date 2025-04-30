# please use Louvain algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result

import networkx as nx
import csv
import random
from typing import List, Set, Dict
from tqdm import tqdm
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
def store_result(G):
    with open('../data/predictions_louvain.csv', 'w') as output:
        output.write("id,category\n")
        for i in range(NUM_NODES):
            output.write("{},{}\n".format(i, G._node[i]['category']))


### TODO ###
### you can define some useful function here if you want

# Code ref: https://github.com/networkx/networkx/blob/main/networkx/algorithms/community/louvain.py

class Louvain:
    def __init__(self, G, random_seed=999):
        """
        Initialize the graph.
        """

        self.G = G
        self.graph = nx.DiGraph()
        random.seed(random_seed)

        node_lists = [(u, {'nodes': {u}}) for u in self.G.nodes()]
        edge_lists = [(u, v, {'weight': 1}) for u, v in self.G.edges()]
        self.graph.add_nodes_from(node_lists)
        self.graph.add_edges_from(edge_lists)

        self.communities = [{u} for u in self.G.nodes()]
        self.edges = self.graph.number_of_edges()


    def modularity(self, node, neighbor, communities: List[Set[int]], belongings: Dict[int,int], 
                   in_degs, out_degs, sum_tot_ins, sum_tot_outs, ngh_weights):
        r"""
        We know the increment of modularity is
            \Delta Q = \dfrac{k_{i,in}}{m} - \dfrac{k_i^{out} \cdot\sum_{tot}^{in} + k_i^{in} \cdot \sum_{tot}^{out}}{m^2}
        """

        k_i_in = in_degs[node]
        k_i_out = out_degs[node]
        
        comm_id = belongings[node]
        nodes = communities[comm_id]
        inside_weight = 0
        for u in nodes:
            inside_weight += ngh_weights[node].get(u, 0)   

        sum_tot_in = sum_tot_ins[comm_id]
        sum_tot_out = sum_tot_outs[comm_id]
        remove_delta_Q = inside_weight / self.edges - (k_i_out * sum_tot_in + k_i_in * sum_tot_out) / (self.edges ** 2)

        ngh_comm = belongings[neighbor]
        nodes = communities[ngh_comm]
        inside_weight = 0
        for u in nodes:
            inside_weight += ngh_weights[node].get(u, 0)
        
        sum_tot_in = sum_tot_ins[ngh_comm]
        sum_tot_out = sum_tot_outs[ngh_comm]
        add_delta_Q = inside_weight / self.edges - (k_i_out * sum_tot_in + k_i_in * sum_tot_out) / (self.edges ** 2)

        return add_delta_Q - remove_delta_Q
    
    def partition(self, G):
        """
        The first phase of Louvain algorithm.
        """

        communities = [{u} for u in G.nodes()]
        belongings = {u: i for i, u in enumerate(self.graph.nodes())}

        in_degs = dict(G.in_degree(weight='weight'))
        out_degs = dict(G.out_degree(weight='weight'))
        sum_tot_ins = list(in_degs.values())
        sum_tot_outs = list(out_degs.values())

        ngh_weights = dict()
        for u in G.nodes():
            if u not in ngh_weights:
                ngh_weights[u] = dict()
            for ngh in G.neighbors(u):
                if ngh not in ngh_weights:
                    ngh_weights[ngh] = dict()
                ngh_weights[u][ngh] = ngh_weights[u].get(ngh, 0) + G[u][ngh]['weight']
                ngh_weights[ngh][u] = ngh_weights[ngh].get(u, 0) + G[u][ngh]['weight']
                    
        visit_seq = list(G.nodes())
        random.shuffle(visit_seq)
        iterations = -1

        while True:
            improvement = False
            iterations += 1
            moves = 0

            for u in tqdm(visit_seq):
                best_delta_Q = 0
                best_comm_id = belongings[u]

                sum_tot_ins[best_comm_id] -= in_degs[u]
                sum_tot_outs[best_comm_id] -= out_degs[u]

                for ngh in G.neighbors(u):
                    delta_Q = self.modularity(u, ngh, communities, belongings, in_degs, out_degs, sum_tot_ins, sum_tot_outs, ngh_weights)
                    if delta_Q > best_delta_Q:
                        best_delta_Q = delta_Q
                        best_comm_id = belongings[ngh]

                sum_tot_ins[best_comm_id] += in_degs[u]
                sum_tot_outs[best_comm_id] += out_degs[u]

                if best_comm_id != belongings[u]:
                    improvement = True
                    communities[belongings[u]].remove(u)
                    communities[best_comm_id].add(u)
                    belongings[u] = best_comm_id
                    moves += 1

            print("Iteration: ", iterations, "\tMoves: ", moves)
            if not improvement:
                break

        communities = [community for community in communities if len(community) > 0]
        return communities, iterations
                

    def restructure(self, G, communities):
        """
        The second phase of Louvain algorithm.
        """

        graph = nx.DiGraph()
        belongings = dict()
        for i in range(len(communities)):
            nodes = set()
            for u in communities[i]:
                belongings[u] = i
                nodes = nodes.union(G.nodes[u]['nodes'])
            graph.add_node(i)
            graph.nodes[i].update({'nodes': nodes})

        for u, v, data in G.edges(data=True):
            weight = data['weight']
            i, j = belongings[u], belongings[v]
            current_edge_weight = graph.get_edge_data(i, j, {'weight': 0})['weight']
            graph.add_edge(i, j)
            graph.edges[i, j].update({'weight': current_edge_weight + weight})

        return graph


    def execute_community(self, communities):
        """
        Assign the community id to each node in the graph.
        """
        for i in range(len(communities)):
            nodes = communities[i]
            for u in nodes:
                for node in self.graph.nodes[u]['nodes']:
                    self.graph.nodes[node].update({'category': i})

        for i in self.graph.nodes():
            if 'category' not in self.graph.nodes[i].keys():
                self.graph.nodes[i].update({'category': -1})

        return self.graph
    
    def execute(self):
        """
        Execute the Louvain algorithm.
        """

        graph = self.graph

        while True:
            print("Partitioning...")
            self.communities, iterations = self.partition(graph)
            if iterations == 0:
                break
            print("Restructuring...")
            graph = self.restructure(graph, self.communities)
            if iterations == 0:
                break
            print("")

        return self.execute_community(self.communities)


### end of TODO ###


def main():
    G = getGraph()

    ### TODO ###
    # implement your community detection alg. here

    louvain = Louvain(G, 20230505)
    G = louvain.execute()

    ### end of TODO ###

    store_result(G)

if __name__ == "__main__":
    main()