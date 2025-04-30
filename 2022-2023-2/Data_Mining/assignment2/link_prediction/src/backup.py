# please use node2vec algorithm to finish the link prediction task
# Do not change the code outside the TODO part

import networkx as nx
import csv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from gensim.models import Word2Vec
# you can use basic operations in networkx
# you can also import other libraries if you need

# read edges.csv and construct the graph
def get_graph():
    G = nx.DiGraph()
    with open("../data/lab2_edges.csv", 'r') as csvFile:
        csvFile.readline()
        csv_reader = csv.reader(csvFile)
        for row in csv_reader:
            source = int(row[0])
            target = int(row[1])
            G.add_edge(source, target)
    print("graph ready")
    return G

# TODO: finish the class Node2Vec
class Node2Vec:
    # you can change the parameters of each function and define other functions
    def __init__(self, graph, num_walks=10, walk_length=80, p=5.0, q=1.0, half_window_size=10, batch_size=512, directed=False):
        self.graph = graph
        self._embeddings = {}
        self._walk_length = walk_length

        self.nodes = list(sorted(self.graph.nodes()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._num_walks = num_walks
        self._p = p
        self._q = q
        self._half_window_size = half_window_size
        self._directed = directed
        self._batch_size = batch_size


    def transition(self, source, target):
        # calculate the transition probability on the graph in the 2-hop neighborhood of the target node

        probs = []
        for v in sorted(self.graph.neighbors(target)):
            if v == source:
                probs.append( 1. / self._p )
            elif self.graph.has_edge(v, source):
                probs.append( 1. )
            else:
                probs.append( 1. / self._q )

        norm_probs = [prob / sum(probs) for prob in probs]
        return norm_probs
    
    
    def alias(self, probs):
        # alias method for sampling from a discrete distribution

        K = len(probs)
        q, J = [0] * K, [0] * K
        smaller, larger = [], []

        for i, prob in enumerate(probs):
            q[i] = K * prob
            if q[i] < 1.0:
                smaller.append(i)
            else:
                larger.append(i)

        while len(smaller) > 0 and len(larger) > 0:
            # pop the last element from the list
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] -= (1.0 - q[small])

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q


    def preprocess(self):
        # prepocess the probabilities in random walk
        # note that J and Q are two events corresponding to the same index in the alias sampling method.

        self._alias_nodes = {}
        self._alias_edges = {}

        for node in self.graph.nodes():
            unnormalized_probs = [1. for v in sorted(self.graph.neighbors(node))]
            norm_probs = [prob / sum(unnormalized_probs) for prob in unnormalized_probs]
            self._alias_nodes[node] = self.alias(norm_probs)

        for edge in self.graph.edges():
            self._alias_edges[edge] = self.alias(self.transition(edge[0], edge[1]))
            if not self._directed:
                self._alias_edges[(edge[1], edge[0])] = self.alias(self.transition(edge[1], edge[0]))

        print("Node2Vec: Preprocession ready.")

        return
    
    
    def sample(self, events):
        # sample from events

        assert len(events) == 2, "events must be a tuple of two lists."
        idx = int(np.floor(np.random.rand() * len(events[0])))
        if np.random.rand() < events[1][idx]:
            return idx
        else:
            return events[0][idx]


    def single_random_walk(self, seed):
        # generate a single random walk of length self._walk_length

        path = [seed]
        while len(path) < self._walk_length:
            node = path[-1]
            neighbors = sorted(self.graph.neighbors(node))

            if len(neighbors) > 0:
                if len(path) == 1:
                    idx = self.sample(self._alias_nodes[node])
                else:
                    prev = path[-2]
                    idx = self.sample(self._alias_edges[(prev, node)])
                path.append(neighbors[idx])
            else:
                break
        
        return path
    

    def _node2onehot(self, node):
        vec = [0] * len(self.graph.nodes())
        vec[self.nodes.index(node)] = 1.
        return vec
    
    def _node2idx(self, node):
        # convert node to index
        return self.nodes.index(node)
    
    def _idx2node(self, idx):
        # convert index to node
        return self.nodes[idx]
    

    def train(self, embed_size=128, epochs=100, neg_spl=10, load_model=None):
        # generate random walks and train node embeddings

        self.preprocess()

        print()
        print("Creating biased walks...")
        nodes = list(self.graph.nodes())
        positive_corpus = []
        for _ in range(self._num_walks):
            np.random.shuffle(nodes)
            xs = []
            ys = []
            for node in tqdm(nodes):
                walk = self.single_random_walk(node)
                for i in range(len(walk)):
                    for j in range(max(i - self._half_window_size, 0), min(i + self._half_window_size + 1, len(walk))):
                        if i != j:
                            xs.append(torch.LongTensor([self._node2idx(walk[i])]))
                            ys.append(torch.LongTensor([self._node2idx(walk[j])]))
                            if len(xs) >= self._batch_size:
                                positive_corpus.append((torch.cat(xs, dim=0), torch.cat(ys, dim=0)))
                                xs = []
                                ys = []

            if len(xs) > 0:
                positive_corpus.append((torch.cat(xs, dim=0), torch.cat(ys, dim=0)))

        negative_corpus=[]
        for batch in tqdm(positive_corpus):
            xs = []
            ys = []
            for i in range(len(batch[0])):
                for _ in range(neg_spl):
                    xs.append(torch.LongTensor([batch[0][i]]))
                    ys.append(torch.LongTensor([np.random.randint(len(self.graph.nodes()))]))
     
            negative_corpus.append((torch.cat(xs, dim=0), torch.cat(ys, dim=0)))

        print("Node2Vec: Biased Walk Dataset ready.")
        print()
        

        # word2vec with skip-gram
        class SkipGram(nn.Module):
            def __init__(self, input_size, emb_size):
                super(SkipGram, self).__init__()
                self._input_size = input_size
                self._emb_size = emb_size
                self.embeddings = nn.Parameter(torch.FloatTensor(input_size, emb_size))
                self.W = nn.Parameter(torch.FloatTensor(input_size, emb_size))
                torch.nn.init.xavier_normal_(self.embeddings)
                torch.nn.init.xavier_normal_(self.W)

            def forward(self, x):
                emb = self.embeddings[x]
                h = emb @ self.W.t()
                return torch.softmax(h, dim=-1)
            

        # train the embedding model.
        # num_training_samples = sum([len(x[0]) for x in positive_corpus])
        self.model = SkipGram(len(self.graph.nodes()), embed_size).to(self.device)
        if load_model is not None:
            self.model.load_state_dict(torch.load(load_model))
            print("Node2Vec: Embedding model loaded.")
            print()
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            loss_fn = nn.NLLLoss()

            print("Node2Vec: Start embedding training...")

            for epoch in range(epochs):

                optimizer.zero_grad()
                epoch_loss = 0

                for i in range(len(positive_corpus)):
                    x = positive_corpus[i][0].to(self.device)
                    label = positive_corpus[i][1].to(self.device)

                    loss = loss_fn(self.model(x), label)
        
                    # negative sampling
                    x = negative_corpus[i][0].to(self.device)
                    label = negative_corpus[i][1].to(self.device)
                    loss -= loss_fn(self.model(x), label)

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if (epoch % 10 == 0) or (epoch == epochs):
                    print("Epoch: {}, Loss: {}".format(epoch, epoch_loss))

            print("Node2Vec: Embedding training completed.")
            torch.save(self.model.state_dict(), "../tmp/model.pt")
            print("Node2Vec: Embedding model saved.")
            print()

        self.model.to("cpu")


    # get embeddings of each node in the graph
    def get_embeddings(self, ):
        for node in self.graph.nodes():
            self._embeddings[self._node2idx(node)] = self.model.embeddings[self._node2idx(node)].detach().numpy()

        return self._embeddings


    # use node embeddings and known edges to train a classifier
    def train_classifier(self, epochs=150, neg_spl=10, load_pairs=None, load_classifier=None):

        class Classifier(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(Classifier, self).__init__()
                self._input_size = input_size
                self._hidden_size = hidden_size
                self._output_size = output_size
                self.fc1 = nn.Linear(input_size, input_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return torch.softmax(x, dim=-1)

        # get embeddings
        if len(self._embeddings.keys()) == 0:
            self._embeddings = self.get_embeddings()

        # construction of training dataset.
        if load_pairs is not None:
            pairs = np.load(load_pairs)
            print("Node2Vec: Training dataset for classifier loaded.")
        else:
            pairs = []
            print("Node2Vec: Creating training dataset for classifier...")
            
            # positive samples
            for edge in self.graph.edges():
                pairs.append((self._node2idx(edge[0]), self._node2idx(edge[1]), 1))
                pairs.append((self._node2idx(edge[1]), self._node2idx(edge[0]), 1))

            # negative samples
            for node in self.graph.nodes():
                for _ in range(neg_spl):
                    neg = np.random.choice(len(self.graph.nodes()))
                    while neg in self.graph.neighbors(node):
                        neg = np.random.choice(len(self.graph.nodes()))
                    pairs.append((self._node2idx(node), neg, 0))

            np.save("../tmp/train_pairs.npy", pairs)
            print("Node2Vec: Training dataset for classifier saved.")
            print()

        # convert to torch tensors
        for idx in self._embeddings.keys():
            self._embeddings[idx] = torch.FloatTensor(self._embeddings[idx])

        np.random.shuffle(pairs)
        self._train_pairs = []
        num_training_samples = int(len(pairs) * 0.6)
        batch = []
        labels = []
        for pair in pairs[:num_training_samples]:
            # batch.append(torch.cat([self._embeddings[pair[0]], self._embeddings[pair[1]]], dim=0).unsqueeze(0)) 
            batch.append( (self._embeddings[pair[0]] - self._embeddings[pair[1]]).unsqueeze(0) )
            labels.append(torch.LongTensor([pair[2]]))
            if len(batch) >= self._batch_size:
                self._train_pairs.append((torch.cat(batch, dim=0), torch.cat(labels, dim=0)))
                batch = []
                labels = []
        self._train_pairs.append((torch.cat(batch, dim=0), torch.cat(labels, dim=0)))

        self._valid_pairs = []
        batch = []
        label = []
        for pair in pairs[num_training_samples:]:
            # batch.append(torch.cat([self._embeddings[pair[0]], self._embeddings[pair[1]]], dim=0).unsqueeze(0)) 
            batch.append( (self._embeddings[pair[0]] - self._embeddings[pair[1]]).unsqueeze(0) )
            labels.append(torch.LongTensor([pair[2]]))
            if len(batch) >= self._batch_size:
                self._valid_pairs.append((torch.cat(batch, dim=0), torch.cat(labels, dim=0)))
                batch = []
                labels = []
        self._valid_pairs.append((torch.cat(batch, dim=0), torch.cat(labels, dim=0)))
        
        # train classifier
        # self._classifier = Classifier(self._embeddings[0].shape[0]*2, 128, 2).to(self.device)
        self._classifier = Classifier(self._embeddings[0].shape[0], 128, 2).to(self.device)
        if load_classifier:
            self._classifier.load_state_dict(torch.load(load_classifier))
            print("Node2Vec: Classifier model loaded.")
            print()
        else:
            print("Node2Vec: Start classifier training...")

            optimizer = torch.optim.Adam(self._classifier.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()
        
            for epoch in range(epochs):
                optimizer.zero_grad()
                epoch_loss = 0

                for pair in self._train_pairs:
                    x = pair[0].to(self.device)
                    label = pair[1].to(self.device)

                    pred = self._classifier(x)
                    loss = loss_fn(pred, label)

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for pair in self._valid_pairs:
                        x = pair[0].to(self.device)
                        label = pair[1].to(self.device)
                        pred = self._classifier(x).argmax().item()
                        correct += (pred == label).sum().item()
                        total += len(label)
                        
                print("Epoch: {}, Loss: {}, Accuracy: {:.4f}".format(epoch, epoch_loss, correct/float(total)))

            print("Node2Vec: Classifier training completed.")
            torch.save(self._classifier.state_dict(), "../tmp/classifier.pt")
            print("Node2Vec: Classifier model saved.")
            print()

        self._classifier.to("cpu")


    def predict(self, source, target):

        if (source not in self.nodes) or (target not in self.nodes):
            return 0

        enc1 = self._embeddings[self._node2idx(source)]
        enc2 = self._embeddings[self._node2idx(target)]

        # use embeddings to predict links
        # prob = self._classifier(torch.cat([torch.tensor(enc1), torch.tensor(enc2)], dim=0))
        prob = self._classifier(torch.tensor( enc1 - enc2 )).detach().numpy()[1]

        return prob

### TODO ###
### you can define some useful functions here if you want


### end of TODO ###

def store_result(model):
    with open('../data/predictions.csv', 'w') as output:
        output.write("id,probability\n")
        with open("../data/lab2_test.csv", 'r') as csvFile:
            csvFile.readline()
            csv_reader = csv.reader(csvFile)
            for row in csv_reader:
                id = int(row[0])
                source = int(row[1])
                target = int(row[2])
                prob = model.predict(source, target)
                output.write("{},{:.4f}\n".format(id, prob))

def main():
    G = get_graph()

    model = Node2Vec(G, num_walks=1)

    # model.train()
    model.train(load_model="../tmp/model.pt")

    embeddings = model.get_embeddings()

    model.train_classifier()
    # model.train_classifier(load_pairs="../tmp/train_pairs.npy", load_classifier="../tmp/classifier.pt")

    store_result(model)

if __name__ == "__main__":
    main()