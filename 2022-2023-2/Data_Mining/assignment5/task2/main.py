import os
import random
import time
from datetime import datetime
import numpy as np
import argparse
from tqdm import tqdm
import torch
from model import CodeNN
from configs import get_config
from data_loader import CodeSearchDataset
from utils import normalize, save_model, get_cosine_schedule_with_warmup
from metrics import ACC, MRR, NDCG
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def train(args):
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    os.makedirs(f'./output/{timestamp}/models', exist_ok=True)
    os.makedirs(f'./output/{timestamp}/tmp_results', exist_ok=True)
    fh = logging.FileHandler(f"./output/{timestamp}/logs.txt")
    logger.addHandler(fh)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    config = get_config()
    print(config)

    ###############################################################################
    # Load data
    ###############################################################################
    train_set = CodeSearchDataset(dataset_path=config['train_data_path'], max_desc_len=config['desc_len'],
                                  max_name_len=config['name_len'], max_tok_len=config['tokens_len'], is_train=True)
    valid_set = CodeSearchDataset(dataset_path=config['valid_data_path'], max_desc_len=config['desc_len'],
                                  max_name_len=config['name_len'], max_tok_len=config['tokens_len'], is_train=True)

    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],
                                              shuffle=True, drop_last=True, num_workers=1)

    ###############################################################################
    # Define Model
    ###############################################################################
    logger.info('Constructing Model..')
    model = CodeNN(config)
    model.to(device)

    ###############################################################################
    # Prepare the Optimizer
    ###############################################################################

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'],
        num_training_steps=len(data_loader) * config['nb_epoch'])

    ###############################################################################
    # Training Process
    ###############################################################################
    TOPK = 10
    n_iters = len(data_loader)
    global_step = 0
    for epoch in range(config['nb_epoch']):
        itr_start_time = time.time()
        losses = []
        for batch in data_loader:
            model.train()
            batch_gpu = [tensor.to(device) for tensor in batch]
            loss = model(*batch_gpu)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            losses.append(loss.item())

            if global_step % args.log_every == 0:
                log_timestamp = datetime.now().strftime('%m-%d-%H-%M')
                elapsed = time.time() - itr_start_time
                logger.info(f'{log_timestamp} epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f' %
                            (epoch, config['nb_epoch'], global_step % n_iters, n_iters, elapsed, np.mean(losses)))
                losses = []
                itr_start_time = time.time()
            global_step = global_step + 1

            if global_step % args.valid_every == 0:
                logger.info("validating..")
                valid_result = validate(valid_set, model, config['batch_size'], 1000, TOPK)
                logger.info(valid_result)

            if global_step % args.save_every == 0:
                ckpt_path = f'./output/{timestamp}/models/step{global_step}.pt'
                save_model(model, ckpt_path)


##### Evaluation #####
def validate(valid_set, model, batch_size, pool_size, TOPK):
    """
    simple validation in a code pool.
    @param: poolsize - size of the code pool, if -1, load the whole test set
    """
    model.eval()
    device = next(model.parameters()).device

    data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size,
                                              shuffle=False, drop_last=True, num_workers=1)
    accs, mrrs, ndcgs = [], [], []
    code_reprs, desc_reprs = [], []
    n_processed = 0
    for batch in tqdm(data_loader):
        # names, name_len, toks, tok_len, descs, desc_len, bad_descs, bad_desc_len
        code_batch = [tensor.to(device) for tensor in batch[:4]]
        desc_batch = [tensor.to(device) for tensor in batch[4:6]]

        with torch.no_grad():
            code_repr = model.code_encoding(*code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32)  # [poolsize x hid_size]
            code_repr = normalize(code_repr)
            desc_repr = normalize(desc_repr)
        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)
    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)

    for k in tqdm(range(0, n_processed, pool_size)):
        code_pool, desc_pool = code_reprs[k:k + pool_size], desc_reprs[k:k + pool_size]
        for i in range(min(len(desc_pool), pool_size)):  # for i in range(pool_size):
            desc_vec = np.expand_dims(desc_pool[i], axis=0)  # [1 x dim]
            n_results = TOPK
            sims = np.dot(code_pool, desc_vec.T)[:, 0]  # [pool_size]

            negsims = np.negative(sims)
            predict = np.argpartition(negsims, kth=n_results - 1)  # predict=np.argsort(negsims)#
            predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict))
            mrrs.append(MRR(real, predict))
            ndcgs.append(NDCG(real, predict))
    return {f'ACC@{TOPK}': np.mean(accs), f'MRR': np.mean(mrrs), f'NDCG@{TOPK}': np.mean(ndcgs)}


def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")

    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--log_every', type=int, default=100, help='interval to log training results')
    parser.add_argument('--valid_every', type=int, default=1000, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000, help='interval to evaluation to concrete results')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
