import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.load_cmpass import load_cmpass,load_engine
import warnings

warnings.filterwarnings('ignore')

class Dataset_CMPASS(Dataset):
    def __init__(self, root_path, flag='train', train_path='train_FD001.txt',test_path='test_FD001.txt',
                 rul_path = 'RUL_FD001.txt',seq_len = 30, engine_num= 0, scale=True):
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.flag = flag

        self.root_path = root_path
        self.train_path = train_path
        self.test_path = test_path
        self.rul_path = rul_path
        self.seq_len = seq_len
        self.__read_data__()

    def __read_data__(self):
        x, y = load_cmpass(self.root_path, self.flag, self.train_path,self.test_path, self.rul_path,self.seq_len, self.scale)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        
        # print(cols)
        if self.flag == 'train' or self.flag == 'val':
            num_train = int(len(x) * 0.95)
            border1s = [0, num_train]
            border2s = [num_train, len(x)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        if self.flag == 'train' or self.flag == 'val':
            self.data_x = x[border1:border2]
            self.data_y = y[border1:border2]
        else:
            self.data_x = x
            self.data_y = y

    def __getitem__(self, index):

        x = self.data_x[index]
        y = self.data_y[index]

        return x, y

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred',train_path='train_FD001.txt',test_path='test_FD001.txt',
                 rul_path = 'RUL_FD001.txt',seq_len = 30, engine_num = 0, scale=True):
        # init
        self.scale = scale

        self.engine_num = engine_num
        self.root_path = root_path
        self.train_path = train_path
        self.test_path = test_path
        self.rul_path = rul_path
        self.seq_len = seq_len

        self.__read_data__()

    def __read_data__(self):
        self.data_x, self.data_y = load_engine(self.engine_num,self.root_path, self.train_path,self.test_path, self.rul_path,self.seq_len, self.scale)


    def __getitem__(self, index):

        seq_x = self.data_x[index]
        seq_y = self.data_y[index]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



    
