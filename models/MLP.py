import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.activation = nn.GELU() if configs.activation == 'gelu' else nn.ReLU()
        self.Linear = nn.Sequential(nn.Flatten(),nn.Linear(configs.seq_len * configs.channel_dim, 1024),
                                    self.activation,nn.Linear(1024, configs.d_ff),
                                    self.activation,nn.Linear(configs.d_ff, configs.pred_len))


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        
        return self.Linear(x) # to [Batch, Output length]
