from os import path
import torch
import torch.utils.data as data
import numpy as np
from utils import indexes2sent, pad_seq
import json
import random


class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """

    def __init__(self, dataset_path: str, max_desc_len: int, max_name_len: int, max_tok_len: int, is_train: False):
        self.max_desc_len = max_desc_len
        self.max_name_len = max_name_len
        self.max_tok_len = max_tok_len

        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)

        self.method_names = []
        self.tokens = []
        self.descs = []
        self.is_train = is_train
        if self.is_train:
            self.bad_descs = []

        for obj in self.dataset:
            method_name = np.asarray(obj['method_name'], dtype=np.longlong)
            self.method_names.append(pad_seq(method_name, self.max_name_len))
            method_token = np.asarray(obj['tokens'], dtype=np.longlong)
            self.tokens.append(pad_seq(method_token, self.max_tok_len))
            method_desc = np.asarray(obj['desc'], dtype=np.longlong)
            self.descs.append(pad_seq(method_desc, self.max_desc_len))
            if self.is_train:
                self.bad_descs.append(pad_seq(method_desc, self.max_desc_len))
        assert len(self.descs) == len(self.method_names)
        assert len(self.method_names) == len(self.tokens)

        self.data_len = len(self.descs)
        print(f'load {self.data_len} samples')
        if self.is_train:
            random.shuffle(self.bad_descs)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.is_train:
            return self.method_names[index], len(self.method_names[index]), self.tokens[index], len(self.tokens[index]), \
                   self.descs[index], len(self.descs[index]), self.bad_descs[index], len(self.bad_descs[index])
        else:
            return self.method_names[index], len(self.method_names[index]), self.tokens[index], len(self.tokens[index])


def load_dict(filename):
    return json.loads(open(filename, "r").readline())


if __name__ == '__main__':
    train_set = CodeSearchDataset(dataset_path='./data/train.json', max_desc_len=30, max_name_len=6, max_tok_len=50,
                                  is_train=True)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=1)

    valid_set = CodeSearchDataset(dataset_path='./data/valid.json', max_desc_len=30, max_name_len=6, max_tok_len=50,
                                  is_train=True)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)

    input_dir = './data'
    vocab_name = load_dict(path.join(input_dir, 'vocab.method_name.json'))
    vocab_tokens = load_dict(path.join(input_dir, 'vocab.tokens.json'))
    vocab_desc = load_dict(path.join(input_dir, 'vocab.description.json'))

    print('============ Train Data ================')
    k = 0
    for batch in train_data_loader:
        batch = tuple([t.numpy() for t in batch])
        name, name_len, tokens, tok_len, good_desc, good_desc_len, bad_desc, bad_desc_len = batch
        k += 1
        if k > 20: break
        print('-------------------------------')
        print(indexes2sent(name, vocab_name))
        print(indexes2sent(tokens, vocab_tokens))
        print(indexes2sent(good_desc, vocab_desc))

    print('\n\n============ Valid Data ================')
    k = 0
    for batch in valid_data_loader:
        batch = tuple([t.numpy() for t in batch])
        name, name_len, tokens, tok_len, good_desc, good_desc_len, bad_desc, bad_desc_len = batch
        k += 1
        if k > 20: break
        print('-------------------------------')
        print(indexes2sent(name, vocab_name))
        print(indexes2sent(tokens, vocab_tokens))
        print(indexes2sent(good_desc, vocab_desc))
