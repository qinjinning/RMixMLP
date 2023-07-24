import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='RUL prediction')

parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--do_predict', type=int, default=0, help='whether to predict RUL of single engine')
#number of engine
parser.add_argument('--engine_num', type=int, default=52, help='number of engine')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='RMixMLP',
                    help='model name, options: [RMixMLP, Linear,MLP]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')

parser.add_argument('--data', type=str, default='FD001', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--train_path', type=str, default='train_FD001.txt', help='data file')
parser.add_argument('--test_path', type=str, default='test_FD001.txt', help='data file')
parser.add_argument('--rul_path', type=str, default='RUL_FD001.txt', help='data file')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=40, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=20, help='prediction sequence length')

#
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
#channel
parser.add_argument('--channel_dim', type=int, default=24, help='channel')
#channel_dim
parser.add_argument('--channel_hidden_dim', type=int, default=128, help='channel_dim')

parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0., help='dropout')


parser.add_argument('--r_times', type=int, default=0, help='loop')
parser.add_argument('--layers', type=int, default=1, help='num of layers')
parser.add_argument('--activation', type=str, default='gelu', help='activation')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='rmse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_rate {}_{}_{}_{}_{}_{}'.format(args.model, args.learning_rate, args.data,args.seq_len,args.pred_len,args.layers,args.r_times)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    # setting record of experiments
    setting = '{}_rate {}_{}_{}_{}_{}_{}'.format(args.model, args.learning_rate, args.data,args.seq_len,args.pred_len,args.layers,args.r_times)


    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
    torch.cuda.empty_cache()
