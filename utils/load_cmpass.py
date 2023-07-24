import torch 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
import os
import numpy as np

# Input files don't have column names
dependent_var = ['RUL']
index_columns_names =  ["ID","Cycle"]
operational_settings_columns_names = ["OpSet"+str(i) for i in range(1,4)]
sensor_measure_columns_names =["Sensor"+str(i) for i in range(1,22)]
input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names


# columns_names = input_file_column_names + ["RUL"]
def acquire_feats(train_path):
    # if train_path == 'train_FD001.txt' or train_path == 'train_FD003.txt':
    #     """FD001&FD003"""
    #     not_required_feats = ["ID","Cycle","OpSet1","OpSet2","OpSet3","Sensor1", "Sensor5", "Sensor6", "Sensor10", "Sensor16", "Sensor18", "Sensor19"]
    # else:
    #     # """FD002&FD004"""
    #     not_required_feats = ["ID","Cycle"]
    not_required_feats = ["ID","Cycle"]
    return  [feat for feat in input_file_column_names if feat not in not_required_feats]


def fun(x):
    if x >= 125:
        return 125
    else:
        return x

def gen_train(id_df, seq_length, seq_cols):
 
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)
    

def gen_target(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        lstm_array.append(data_array[start:stop])
    return np.array(lstm_array)

    # data_array = id_df[label].values
    # num_elements = data_array.shape[0]
    # return data_array[seq_length-1:num_elements+1]


def gen_test(id_df, seq_length, seq_cols):
    df_mask = pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    for col in df_mask.columns:
        df_mask[col] = id_df[col].mean()
    # df_mask[:] = 0
    id_df = df_mask.append(id_df,ignore_index=True)
    
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]

    start = num_elements-seq_length
    stop = num_elements
    
    lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)

def load_cmpass(root_path='/home/jinn/pdm/RUL/dataset', flag='train', train_path='train_FD001.txt', test_path = 'test_FD001.txt' ,rul_path = None,
sequence_length = 36,scale = True):
    df_train = pd.read_csv(os.path.join(root_path, train_path),delim_whitespace=True,names=input_file_column_names)
    df_test = pd.read_csv(os.path.join(root_path, test_path),delim_whitespace=True,names=input_file_column_names)
    rul = pd.DataFrame(df_train.groupby('ID')['Cycle'].max()).reset_index()
    rul.columns = ['ID', 'max']
    df_train = df_train.merge(rul, on=['ID'], how='left')
    df_train['RUL'] = df_train['max'] - df_train['Cycle']
    df_train.drop('max', axis=1, inplace=True)
    df_train['RUL'] = df_train['RUL'].apply(lambda x: fun(x))
    if flag == 'test':
        rul = pd.DataFrame(df_test.groupby('ID')['Cycle'].max()).reset_index()
        rul.columns = ['ID', 'max']
        df_test = df_test.merge(rul, on=['ID'], how='left')
        df_test['RUL'] = df_test['max'] - df_test['Cycle']
        df_test.drop('max', axis=1, inplace=True)
        df_rul = pd.read_csv(os.path.join(root_path, rul_path),delim_whitespace=True,names=['RUL'])
        df_rul["ID"] = df_rul.index+1
        actual_rul = pd.DataFrame(df_rul.groupby('ID')['RUL'].max()).reset_index()
        actual_rul.columns = ['ID', 'acrul']
        df_test = df_test.merge(actual_rul, on=['ID'], how='left')
        df_test['RUL'] = df_test['RUL']+df_test['acrul']
        df_test.drop('acrul', axis=1, inplace=True)
        df_test['RUL']=df_test['RUL'].apply(lambda x: fun(x))
        df_rul['RUL']=df_rul['RUL'].apply(lambda x: fun(x))
    
    feats = acquire_feats(train_path)
    if scale:
        min_max_scaler = MinMaxScaler(feature_range=(0,1))
        df_train[feats] = min_max_scaler.fit_transform(df_train[feats])
        df_test[feats] = min_max_scaler.transform(df_test[feats])
    #generate train
    if flag == 'test':
        x = np.concatenate(list(list(gen_test(df_test[df_test['ID']==unit], sequence_length, feats)) 
                           for unit in df_test['ID'].unique()))
        
        y = torch.tensor(df_rul.RUL.values)
    else:
        x = np.concatenate(list(list(gen_train(df_train[df_train['ID']==unit], sequence_length, feats)) 
                                        for unit in df_train['ID'].unique()))
        #generate target of train
        y = torch.tensor(np.concatenate(list(list(gen_target(df_train[df_train['ID']==unit], sequence_length, "RUL")) 
                                for unit in df_train['ID'].unique())))
    return x, y
  
def load_engine(unit=1, root_path='/home/jinn/pdm/RUL/dataset', train_path='train_FD001.txt', test_path = 'test_FD001.txt' ,rul_path = 'RUL_FD001.txt',
sequence_length = 36,scale = True):
    df_train = pd.read_csv(os.path.join(root_path, train_path),delim_whitespace=True,names=input_file_column_names)
    df_test = pd.read_csv(os.path.join(root_path, test_path),delim_whitespace=True,names=input_file_column_names)


    rul = pd.DataFrame(df_test.groupby('ID')['Cycle'].max()).reset_index()
    rul.columns = ['ID', 'max']
    df_test = df_test.merge(rul, on=['ID'], how='left')
    df_test['RUL'] = df_test['max'] - df_test['Cycle']
    df_test.drop('max', axis=1, inplace=True)
    df_rul = pd.read_csv(os.path.join(root_path, rul_path),delim_whitespace=True,names=['RUL'])
    df_rul["ID"] = df_rul.index+1
    actual_rul = pd.DataFrame(df_rul.groupby('ID')['RUL'].max()).reset_index()
    actual_rul.columns = ['ID', 'acrul']
    df_test = df_test.merge(actual_rul, on=['ID'], how='left')
    df_test['RUL'] = df_test['RUL']+df_test['acrul']
    df_test.drop('acrul', axis=1, inplace=True)
    df_test['RUL']=df_test['RUL'].apply(lambda x: fun(x))
    df_rul['RUL']=df_rul['RUL'].apply(lambda x: fun(x))

    feats = acquire_feats(train_path)
    if scale:
        min_max_scaler = MinMaxScaler(feature_range=(0,1))
        df_train[feats] = min_max_scaler.fit_transform(df_train[feats])
        df_test[feats] = min_max_scaler.transform(df_test[feats])
    #generate train
    
    x = np.concatenate(list(list(gen_train(df_test[df_test['ID']==unit], sequence_length, feats)) 
                        for unit in [unit]))
    
    y = torch.tensor(np.concatenate(list(list(gen_target(df_test[df_test['ID']==unit], sequence_length, "RUL")) 
                            for unit in [unit])))

    return x, y