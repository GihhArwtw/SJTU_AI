import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BOWEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(BOWEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size=emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        
    def forward(self, input): 
        seq_len = input.size()[1]
        embedded = self.embedding(input)                    # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        embedded= F.dropout(embedded, 0.25, self.training)  # [batch_size x seq_len x emb_size]
        
        # max pooling word vectors
        output_pool = F.max_pool1d(embedded.transpose(1,2), seq_len).squeeze(2)         # [batch_size x emb_size]
        encoding = output_pool                              # torch.tanh(output_pool)
        return encoding
    

class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name or 'bias' in name: 
                param.data.uniform_(-0.1, 0.1)

    def forward(self, inputs, input_lens=None): 
        batch_size = inputs.size()[0]
        inputs = self.embedding(inputs)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        inputs = F.dropout(inputs, 0.25, self.training)
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
        hids, (h_n, c_n) = self.lstm(inputs) # hids:[b x seq x hid_sz*2](biRNN) 
        
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)   
            hids = F.dropout(hids, p=0.25, training=self.training)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, 2, batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] 
        encoding = h_n.view(batch_size,-1)

        return encoding 


class CodeNN(nn.Module):
    def __init__(self, config):
        super(CodeNN, self).__init__()
        self.conf = config
        self.margin = config['margin']
        self.name_encoder = SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        self.token_encoder = BOWEncoder(config['n_words'],config['emb_size'])
        self.desc_encoder = SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        
        self.w_name = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.w_tok = nn.Linear(config['emb_size'], config['n_hidden'])
        self.w_desc = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.fuse = nn.Linear(config['n_hidden'], config['n_hidden'])

    def code_encoding(self, name, name_len, tokens, token_len):
        name_repr = self.name_encoder(name, name_len)
        token_repr = self.token_encoder(tokens)
        code_repr = self.fuse(torch.tanh(self.w_name(name_repr) + self.w_tok(token_repr)))
        
        return code_repr
    
    def desc_encoding(self, desc, desc_len):
        desc_repr = self.desc_encoder(desc, desc_len)
        desc_repr = self.w_desc(desc_repr)
        
        return desc_repr
    
    def similarity(self, code_vec, desc_vec):
        return F.cosine_similarity(code_vec, desc_vec)

    def forward(self, name, name_len, tokens, token_len, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        code_repr = self.code_encoding(name, name_len, tokens, token_len)
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        neg_sim = self.similarity(code_repr, desc_neg_repr)  # [batch_sz x 1]

        loss = (self.margin - anchor_sim + neg_sim).clamp(min=1e-6).mean()

        return loss
