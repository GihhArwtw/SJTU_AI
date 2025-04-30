import numpy as np
import time
import os
import math
import torch
from torch.nn import functional as F

PAD_ID, SOS_ID, EOS_ID, UNK_ID = [0, 1, 2, 3]


def pad_seq(seq, maxlen):
    if len(seq) < maxlen:
        # !!!!! numpy appending is slow. Try to optimize the padding
        seq = np.append(seq, [PAD_ID] * (maxlen - len(seq)))
    seq = seq[:maxlen]
    return seq

#######################################################################

def indexes2sent(indexes, vocab, ignore_tok=PAD_ID):
    '''indexes: numpy array'''

    def revert_sent(indexes, ivocab, ignore_tok=PAD_ID):
        indexes = filter(lambda i: i != ignore_tok, indexes)
        toks, length = [], 0
        for idx in indexes:
            toks.append(ivocab.get(idx, '<unk>'))
            length += 1
            if idx == EOS_ID:
                break
        return ' '.join(toks), length

    ivocab = {v: k for k, v in vocab.items()}
    if indexes.ndim == 1:  # one sentence
        return revert_sent(indexes, ivocab, ignore_tok)
    else:  # dim>1
        sentences, lens = [], []  # a batch of sentences
        for inds in indexes:
            sentence, length = revert_sent(inds, ivocab, ignore_tok)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens

########################################################################

def save_model(model, ckpt_path):
    torch.save(model.state_dict(), ckpt_path)


from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def normalize(data):
    """normalize matrix by rows"""
    return data/np.linalg.norm(data,axis=1,keepdims=True)