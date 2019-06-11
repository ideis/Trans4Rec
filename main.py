import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from trans4rec import Trans4Rec
from dataset import load_data
from loss import WARPLoss
from evaluation import evaluate_scores

def train():
    model.train()
    train_loss = list()
    for batch_id, batch in enumerate(train_generator):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = (src.transpose(1, 0) == 0)
        preds = model(src, src_mask, shared_embeddings=True)

        if loss_function == 'warp':
            preds = preds[-1, :, :]
            y = torch.zeros(batch_size, n_items, dtype=torch.long)
            y[:, tgt.view(-1)] = 1
            loss = criterion(preds, tgt)
        else:
            loss = criterion(preds.contiguous().view(-1, n_items), tgt.contiguous().view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
        if batch_id % logs == 0 and batch_id != 0:
            print(f'TRAIN | epoch: {epoch} |  samples: {batch_id}/{len(train_generator)} | loss: {np.mean(train_loss):.4f}')

def evaluate():
    model.eval()
    eval_loss = list()
    mrrs = list()
    recalls_20 = list()
    total_covered_items = set()
    for batch in test_generator:
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)    
        src_mask = (src.transpose(1, 0) == 0)
        preds = model(src, src_mask, shared_embeddings=True)

        if loss_function == 'warp':
            preds = preds[-1, :, :]
            y = torch.zeros(batch_size, n_items, dtype=torch.long)
            y[:, tgt.view(-1)] = 1
            loss = criterion(preds, tgt)
        else:
            loss = criterion(preds.contiguous().view(-1, n_items), tgt.contiguous().view(-1))

        eval_loss.append(loss.item())
        mrr, recall_10, covered_items = evaluate_scores(preds, tgt, k=10)
        mrrs.append(mrr)
        recalls_10.append(recall_10)
        total_covered_items.update(covered_items)
    return eval_loss, mrrs, recalls_10, total_covered_items

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'device: {device}')

# Learning hyperparameters
batch_size = 32
epochs = 20
lr = 0.001
logs = 100
loss_function = 'warp'

#Transformer hyperparameters
d_model = 256
nhead = 2
num_encoder_layers = 4
dim_feedforward = 512
dropout = 0.1

model = Trans4Rec(n_items, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
model.to(device)

if loss_function == 'warp':
    criterion = WARPLoss(max_num_trials=1000)
else:
    criterion = nn.CrossEntropyLoss().to(device)

optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

datasets = ['taobao_buy', 'taobao_cart', 'tmall_clicks']
for dataset in datasets:    
    train_generator, test_generator, n_items = load_data('data/{dataset}', batch_size)
    try:
        for epoch in range(1, epochs+1):
            train()
            print('-' * 89)
            eval_loss, mrrs, recalls_10, total_covered_items = evaluate()
            print(f'EVAL | epoch: {epoch} | loss: {np.mean(eval_loss):.4f}')
            print(f'MRR: {np.mean(mrrs):.5f} | R@10: {np.mean(recalls_10):.5f} | Coverage: {len(total_covered_items)/n_items}')
            print('-' * 89)
            # Save the model at each 5 epoch.
            with open('model'+ str(epoch) +'.pt', 'wb') as f:
                torch.save(model, f)

    except KeyboardInterrupt:
        print('-' * 89)
        print('exiting from training early')