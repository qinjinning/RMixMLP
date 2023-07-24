import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RMixMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, in_channel,hidden_channel,out_channel,activation,dropout_rate=0.):
        super(RMixMLPBlock, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channel, out_channel),
            nn.Dropout(dropout_rate)
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            activation,
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.Dropout(dropout_rate)
        )
        self.layernorm = nn.LayerNorm(out_channel)


    def forward(self, x):
        # x: [B,in_channel,in_dim] 
        h1 = self.net1(x)  # [B,in_channel,in_dim] -> [B,in_channel,out_dim]
        h2 = self.net2(x)  # [B,in_channel,in_dim] -> [B,out_channel,in_dim]
        h3 = self.net2(h1)  # [B,in_channel,in_dim] -> [B,out_channel,in_dim]
        h4 = self.net1(h2)  # [B,in_channel,in_dim] -> [B,in_channel,out_dim]
        out = h1 + h2 + h3 + h4
        out = self.layernorm(out)  # [B,in_channel,out_dim] + [B,in_channel,in_dim] -> [B,in_channel,out_dim]
        return out
    
class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.layers = configs.layers
        self.r_times = configs.r_times
        self.activation = nn.GELU() if configs.activation == 'gelu' else nn.ReLU()
        self.RMixMLPBlock_layers = nn.ModuleList([
            RMixMLPBlock(configs.seq_len, configs.hidden_dim, configs.seq_len, configs.channel_dim, configs.channel_hidden_dim, configs.channel_dim,self.activation,configs.dropout) for _ in range(self.layers)
            ])
        self.Linear = nn.Sequential(nn.Flatten(),nn.Linear(configs.seq_len * configs.channel_dim, configs.pred_len))


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        
        for i in range(self.layers):
            for _ in range(self.r_times+1):
                x = self.RMixMLPBlock_layers[0](x)
        x = self.Linear(x) 
        return x # [Batch, Output length, Channel]