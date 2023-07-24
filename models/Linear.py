import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Sequential(nn.Flatten(),nn.Linear(configs.seq_len * configs.channel_dim, configs.pred_len))

    def forward(self, x):

        return self.Linear(x)  # [Batch, Output length, Channel]