# please use node2vec algorithm to finish the link prediction task
# Do not change the code outside the TODO part

import networkx as nx
import csv
import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
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

# class word2vec():

# 	def __init__(self):
# 		self.n = settings['n']
# 		self.lr = settings['learning_rate']
# 		self.epochs = settings['epochs']
# 		self.window = settings['window_size']

# 	def generate_training_data(self, settings, corpus):
# 		# Find unique word counts using dictonary
# 		word_counts = defaultdict(int)
# 		for row in corpus:
# 			for word in row:
# 				word_counts[word] += 1
# 		#########################################################################################################################################################
# 		# print(word_counts)																																	#
# 		# # defaultdict(<class 'int'>, {'natural': 1, 'language': 1, 'processing': 1, 'and': 2, 'machine': 1, 'learning': 1, 'is': 1, 'fun': 1, 'exciting': 1})	#
# 		#########################################################################################################################################################

# 		## How many unique words in vocab? 9
# 		self.value_count = len(word_counts.keys())
# 		#########################
# 		# print(self.value_count)	#
# 		# 9						#
# 		#########################

# 		# Generate Lookup Dictionaries (vocab)
# 		self.words_list = list(word_counts.keys())
# 		#################################################################################################
# 		# print(self.words_list)																		#
# 		# ['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'exciting']	#
# 		#################################################################################################
		
# 		# Generate word:index
# 		self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
# 		#############################################################################################################################
# 		# print(self.word_index)																									#
# 		# # {'natural': 0, 'language': 1, 'processing': 2, 'and': 3, 'machine': 4, 'learning': 5, 'is': 6, 'fun': 7, 'exciting': 8}	#
# 		#############################################################################################################################

# 		# Generate index:word
# 		self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
# 		#############################################################################################################################
# 		# print(self.index_word)																									#
# 		# {0: 'natural', 1: 'language', 2: 'processing', 3: 'and', 4: 'machine', 5: 'learning', 6: 'is', 7: 'fun', 8: 'exciting'}	#
# 		#############################################################################################################################

# 		training_data = []

# 		# Cycle through each sentence in corpus
# 		for sentence in corpus:
# 			sent_len = len(sentence)

# 			# Cycle through each word in sentence
# 			for i, word in enumerate(sentence):
# 				# Convert target word to one-hot
# 				w_target = self.word2onehot(sentence[i])

# 				# Cycle through context window
# 				w_context = []

# 				# Note: window_size 2 will have range of 5 values
# 				for j in range(i - self.window, i + self.window+1):
# 					# Criteria for context word 
# 					# 1. Target word cannot be context word (j != i)
# 					# 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
# 					# 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range 
# 					if j != i and j <= sent_len-1 and j >= 0:
# 						# Append the one-hot representation of word to w_context
# 						w_context.append(self.word2onehot(sentence[j]))
# 						# print(sentence[i], sentence[j]) 
# 						#########################
# 						# Example:				#
# 						# natural language		#
# 						# natural processing	#
# 						# language natural		#
# 						# language processing	#
# 						# language append 		#
# 						#########################
						
# 				# training_data contains a one-hot representation of the target word and context words
# 				#################################################################################################
# 				# Example:																						#
# 				# [Target] natural, [Context] language, [Context] processing									#
# 				# print(training_data)																			#
# 				# [[[1, 0, 0, 0, 0, 0, 0, 0, 0], [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]]]	#
# 				#################################################################################################
# 				training_data.append([w_target, w_context])

# 		return np.array(training_data)

# 	def word2onehot(self, word):
# 		# word_vec - initialise a blank vector
# 		word_vec = [0 for i in range(0, self.value_count)] # Alternative - np.zeros(self.value_count)
# 		#############################
# 		# print(word_vec)			#
# 		# [0, 0, 0, 0, 0, 0, 0, 0]	#
# 		#############################

# 		# Get ID of word from word_index
# 		word_index = self.word_index[word]

# 		# Change value from 0 to 1 according to ID of the word
# 		word_vec[word_index] = 1

# 		return word_vec

# 	def train(self, training_data):

# 		self.w1 = np.array(getW1)
# 		self.w2 = np.array(getW2)

		
# 		# Cycle through each epoch
# 		for i in range(self.epochs):
# 			# Intialise loss to 0
# 			self.loss = 0
# 			# Cycle through each training sample
# 			# w_t = vector for target word, w_c = vectors for context words
# 			for w_t, w_c in training_data:
# 				# Forward pass
# 				# 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
# 				y_pred, h, u = self.forward_pass(w_t)
# 				#########################################
# 				# print("Vector for target word:", w_t)	#
# 				# print("W1-before backprop", self.w1)	#
# 				# print("W2-before backprop", self.w2)	#
# 				#########################################

# 				# Calculate error
# 				# 1. For a target word, calculate difference between y_pred and each of the context words
# 				# 2. Sum up the differences using np.sum to give us the error for this particular target word
# 				EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
# 				#########################
# 				# print("Error", EI)	#
# 				#########################

# 				# Backpropagation
# 				# We use SGD to backpropagate errors - calculate loss on the output layer 
# 				self.backprop(EI, h, w_t)
# 				#########################################
# 				#print("W1-after backprop", self.w1)	#
# 				#print("W2-after backprop", self.w2)	#
# 				#########################################

# 				# Calculate loss
# 				# There are 2 parts to the loss function
# 				# Part 1: -ve sum of all the output +
# 				# Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
# 				# Note: word.index(1) returns the index in the context word vector with value 1
# 				# Note: u[word.index(1)] returns the value of the output layer before softmax
# 				self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
				
# 				#############################################################
# 				# Break if you want to see weights after first target word 	#
# 				# break 													#
# 				#############################################################
# 			print('Epoch:', i, "Loss:", self.loss)

# 	def forward_pass(self, x):
# 		# x is one-hot vector for target word, shape - 9x1
# 		# Run through first matrix (w1) to get hidden layer - 10x9 dot 9x1 gives us 10x1
# 		h = np.dot(x, self.w1)
# 		# Dot product hidden layer with second matrix (w2) - 9x10 dot 10x1 gives us 9x1
# 		u = np.dot(h, self.w2)
# 		# Run 1x9 through softmax to force each element to range of [0, 1] - 1x8
# 		y_c = self.softmax(u)
# 		return y_c, h, u

# 	def softmax(self, x):
# 		e_x = np.exp(x - np.max(x))
# 		return e_x / e_x.sum(axis=0)

# 	def backprop(self, e, h, x):
# 		# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
# 		# Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
# 		# Going backwards, we need to take derivative of E with respect of w2
# 		# h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
# 		# x - shape 9x1, w2 - 10x9, e.T - 9x1
# 		dl_dw2 = np.outer(h, e)
# 		dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
# 		########################################
# 		# print('Delta for w2', dl_dw2)			#
# 		# print('Hidden layer', h)				#
# 		# print('np.dot', np.dot(self.w2, e.T))	#
# 		# print('Delta for w1', dl_dw1)			#
# 		#########################################

# 		# Update weights
# 		self.w1 = self.w1 - (self.lr * dl_dw1)
# 		self.w2 = self.w2 - (self.lr * dl_dw2)

# 	# Get vector from word
# 	def word_vec(self, word):
# 		w_index = self.word_index[word]
# 		v_w = self.w1[w_index]
# 		return v_w

# 	# Input vector, returns nearest word(s)
# 	def vec_sim(self, word, top_n):
# 		v_w1 = self.word_vec(word)
# 		word_sim = {}

# 		for i in range(self.value_count):
# 			# Find the similary score for each word in vocab
# 			v_w2 = self.w1[i]
# 			theta_sum = np.dot(v_w1, v_w2)
# 			theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
# 			theta = theta_sum / theta_den

# 			word = self.index_word[i]
# 			word_sim[word] = theta

# 		words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

# 		for word, sim in words_sorted[:top_n]:
# 			print(word, sim)

class Node2Vec:
    # you can change the parameters of each function and define other functions
    # def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, ):
    #     self.graph = graph
    #     self._embeddings = {}
    #     self.walk_length = walk_length
    #     self.num_walks = num_walks
    #     self.p = p
    #     self.q = q
        
        
    def __init__(self, graph, walk_length=80, num_walks=10, p=5.0, q=1.0, window_size=10):
        self.graph = graph
        self._embeddings = {}
        self._classifier = None
        self.edges_info = {}
        self.nodes_info = {}
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.model = None
        
    def get_alias_nodes(self, probs):
        l = len(probs)
        a, b = np.zeros(l), np.zeros(l, dtype=np.int)
        smaller, larger = [], []

        for i, prob in enumerate(probs):
            a[i] = l * prob
            if a[i] < 1.0:
                smaller.append(i)
            else:
                larger.append(i)
            
        while smaller and larger:
            small, large = smaller.pop(), larger.pop()
            b[small] = large
            a[large] += a[small] - 1.0
            if a[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
                
        return b, a
        
    def get_alias_edges(self, source, target):
        probs = []
        for v in sorted(self.graph.neighbors(target)):
            if v == source:
                probs.append(1 / self.p)
            elif self.graph.has_edge(v, source):
                probs.append(1)
            else:
                probs.append(1 / self.q)
        norm_probs = [float(prob) / sum(probs) for prob in probs]
        return self.get_alias_nodes(norm_probs)
        
    def preprocess(self):
        alias_nodes, alias_edges = {}, {}
        for u in self.graph.nodes():
            probs = [1 for v in sorted(self.graph.neighbors(u))]
            norm_const = sum(probs)
            norm_probs = [float(prob) / norm_const for prob in probs]
            alias_nodes[u] = self.get_alias_nodes(norm_probs)
    
        for edge in self.graph.edges():
            alias_edges[edge] = self.get_alias_edges(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edges(edge[1], edge[0])

        return alias_nodes, alias_edges
    
    def node2vec_walk(self, start, alias_nodes, alias_edges):
        path = [start]
        while len(path) < self.walk_length:
            node = path[-1]
            neighbors = sorted(self.graph.neighbors(node))
            if len(neighbors) > 0:
                if len(path) == 1:
                    l = len(alias_nodes[node][0])
                    idx = int(np.floor(np.random.rand() * l))
                    if np.random.rand() < alias_nodes[node][1][idx]:
                        path.append(neighbors[idx])
                    else:
                        path.append(neighbors[alias_nodes[node][0][idx]])
                else:
                    prev = path[-2]
                    l = len(alias_edges[(prev, node)][0])
                    idx = int(np.floor(np.random.rand() * l))
                    if np.random.rand() < alias_edges[(prev, node)][1][idx]:
                        path.append(neighbors[idx])
                    else:
                        path.append(neighbors[alias_edges[(prev, node)][0][idx]])
            else:
                break
        return path 

    def train(self, embed_size=128):
        walks = []
        alias_nodes, alias_edges = self.preprocess()
        nodes = list(self.graph.nodes())
        for t in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vec_walk(node, alias_nodes, alias_edges)
                walks.append(walk)
        # embedding
        walks = [list(map(str, walk)) for walk in walks]
        print("Training word2vec...")
        self.model = Word2Vec(sentences=walks, vector_size=embed_size, window=self.window_size, min_count=0, sg=1, workers=3, epochs=3)
        print("training complete.")

    # get embeddings of each node in the graph
    def get_embeddings(self, ):
        for node in self.graph.nodes():
            self._embeddings[node] = self.model.wv[str(node)]

        return self._embeddings

    def generate_data(self):
        edges, nodes = list(self.graph.edges()), list(self.graph.nodes())
        positive_samples, negative_samples = [], []

        for source, target in tqdm(edges):
            positive_samples.append((source, target))
            while True:
                flag = False
                negative_source = np.random.choice(nodes)
                negative_target = np.random.choice(nodes)
                sample1 = (negative_source, target)
                sample2 = (source, negative_target)
                sample3 = (negative_source, negative_target)
                if sample1 not in edges and sample1 not in negative_samples:
                    negative_samples.append(sample1)
                    flag = True
                if sample2 not in edges and sample2 not in negative_samples:
                    negative_samples.append(sample2)
                    flag = True
                if sample3 not in edges and sample3 not in negative_samples:
                    negative_samples.append(sample3)
                    flag = True
                if flag:
                    break

        positive_labels = [1 for _ in range(len(positive_samples))]
        negative_labels = [0 for _ in range(len(negative_samples))]
        labels = positive_labels + negative_labels
        samples = positive_samples + negative_samples
        datas = [self._embeddings[target] - self._embeddings[source] for source, target in samples]

        indices = np.arange(len(datas))
        np.random.shuffle(indices)
        datas = np.array(datas)[indices]
        labels = np.array(labels)[indices]

        train_size = int(0.8 * len(datas))
        train_datas, train_labels = datas[:train_size], labels[:train_size]
        val_datas, val_labels = datas[train_size:], labels[train_size:]
        np.save('../data/train_datas.npy', train_datas)
        np.save('../data/train_labels.npy', train_labels)
        np.save('../data/val_datas.npy', val_datas)
        np.save('../data/val_labels.npy', val_labels)
        return train_datas, train_labels, val_datas, val_labels


    # use node embeddings and known edges to train a classifier
    def train_classifier(self):
        generate_data = True
        if generate_data:
            train_datas, train_labels, val_datas, val_labels = self.generate_data()        
        else:
            train_datas = np.load('../data/train_datas.npy')
            train_labels = np.load('../data/train_labels.npy')
            val_datas = np.load('../data/val_datas.npy')
            val_labels = np.load('../data/val_labels.npy')
            
        class Classifier(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                x = F.softmax(x, dim=1)
                return x
        # train
        is_trained = False
        if not is_trained:
            print('start training')
            classifier = Classifier(128, 64, 2)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(classifier.parameters(), lr=0.001)
            epochs = 10
            for epoch in range(epochs):
                pbar = tqdm(range(len(train_datas)))
                loss_sum = 0.0
                # shuffle train datas
                indices = np.arange(len(train_datas))
                np.random.shuffle(indices)
                train_datas = train_datas[indices]
                train_labels = train_labels[indices]
                for i in pbar:
                    optimizer.zero_grad()
                    output = classifier(torch.Tensor([train_datas[i]]))
                    loss = criterion(output, torch.LongTensor([train_labels[i]]))
                    loss.backward()
                    loss_sum += loss.item()
                    optimizer.step()
                    pbar.set_postfix({'epoch': epoch, 'avg_loss': loss_sum / (i+1)})
                # evaluate
                correct = 0
                total = 0
                with torch.no_grad():
                    for i in range(len(val_datas)):
                        output = classifier(torch.Tensor([val_datas[i]]))
                        predict = output.argmax().item()
                        correct += (predict == val_labels[i])
                        total += 1
                print('Accuracy of the network on the validation set: %.2f %%' % (100.0 * correct / total))
            # save model
            torch.save(classifier.state_dict(), '../data/classifier.pt')
        else:
            print('loading trained model')
            classifier = Classifier(128, 64, 2)
            classifier.load_state_dict(torch.load('../data/classifier.pt'))
        self._classifier = classifier


    def predict(self, source, target):
        if source not in self._embeddings or target not in self._embeddings:
            return 0
        enc1 = self._embeddings[source]
        enc2 = self._embeddings[target]
        input = torch.Tensor(np.array([enc2 - enc1]))
        # use embeddings to predict links
        prob = self._classifier(input).detach().numpy()[0][1]

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

    model = Node2Vec(G, )

    model.train()

    embeddings = model.get_embeddings()

    model.train_classifier()

    store_result(model)

if __name__ == "__main__":
    main()