import random
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class AliDataset(Dataset):
    def __init__(self, data, n_items):
        self.data = data
        self.n_items = n_items
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        X = torch.LongTensor(self.data[index][:-1])
        y = torch.LongTensor(self.data[index][1:])
        return X, y

class PadCollator(object):
    '''
    Pad sequences with 0 by max length in batch
    '''
    def __call__(self, batch):
        X = [item[0] for item in batch]
        y = [item[1] for item in batch]
        X = pad_sequence(X, batch_first=False)
        y = pad_sequence(y, batch_first=True)
        return X, y

def load_data(dataset, batch_size=1):

    train_data = pd.read_csv(f'{dataset}_train.csv', sep='\t')
    train = train_data.groupby('UserId')['ItemId'].apply(list).to_dict()
    n_items = train_data.ItemId.nunique()

    test_data = pd.read_csv(f'{dataset}_test.csv', sep='\t')
    test = test_data.groupby('UserId')['ItemId'].apply(list).to_dict()

    train = AliDataset(train, n_items)
    test = AliDataset(test, n_items)
    collate = PadCollator()

    train_generator = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16, collate_fn=collate)
    test_generator = DataLoader(test, batch_size=1, shuffle=True, drop_last=True, num_workers=16, collate_fn=collate)
    return train_generator, test_generator, n_items
