from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import RMixMLP,Linear,MLP
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.lossfunction import LogCoshLoss,rmse_loss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'RMixMLP':RMixMLP,
            'Linear':Linear,
            'MLP':MLP

        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'rmse':
            criterion = rmse_loss()
        elif self.args.loss == 'SmoothL1Loss':
            criterion = nn.SmoothL1Loss()
        elif self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'logcosh':
            criterion = LogCoshLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion,falg='train'):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)

                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, :, f_dim:].squeeze(2)
                pred = outputs.detach().cpu()

                
                true = batch_y.detach().cpu()
                if falg=='test':
                    pred = pred[:, -1]
                else:
                    true = batch_y[:, -self.args.pred_len:].detach().cpu()
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # 记录时间
        time_now = time.time()

        # 训练steps
        train_steps = len(train_loader)
        # 早停策略
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 优化器
        model_optim = self._select_optimizer()
        # 损失函数(MSE)
        criterion = self._select_criterion()

        # 分布式训练(windows一般不推荐)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 训练次数
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                # 梯度归零
                model_optim.zero_grad()
                # 取训练数据
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                # f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                # 计算损失
                loss = criterion(outputs, batch_y)
                # 将损失放入train_loss列表中
                train_loss.append(loss.item())
                # 记录训练过程
                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 反向传播
                    loss.backward()
                    # 更新梯度
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion, 'test')

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # 更新学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        # 保存模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),map_location=torch.device('cuda:{}'.format(self.args.gpu))))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs[:,-1]  
                true = batch_y  
                preds.append(pred)
                trues.append(true)


        # preds = np.array(np.concatenate(preds))
        # trues = np.array(np.concatenate(trues))
        preds = np.array(preds)
        trues = np.array(trues)
        index = np.array(range(1, preds.shape[0] + 1))

        dfData = {
        'index':index,
        'preds':preds.ravel(),
        'trues':trues.ravel()}
        df = pd.DataFrame(dfData)
        df.to_csv(os.path.join(folder_path, setting + '.csv'),index=False)

        visual(trues, preds, os.path.join(folder_path, setting + '.pdf'))
        
        # f = open("pred.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('preds:{}, trues:{}'.format(preds.ravel(), trues.ravel()))
        # f.write('\n')
        # f.write('\n')
        # f.close()



        mae, score, rmse = metric(preds, trues)
        print('score:{}, mae:{}, rmse:{}'.format(score, mae, rmse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('score:{}, mae:{}, rmse:{}'.format(score, mae, rmse))
        f.write('\n')
        f.write('\n')
        f.close()
        return
    

    def predict(self, setting, predict=0):
        predict_data, predict_loader = self._get_data(flag='pred')
        
        if predict:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = [125]*self.args.seq_len
        trues = [125]*self.args.seq_len
        folder_path = './predict_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(predict_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, :, f_dim:].squeeze(2)
                
                pred = outputs[:, -self.args.pred_len:].squeeze(0)
                true = batch_y[:, -self.args.pred_len:].squeeze(0)
    

                preds.append(pred[0])
                trues.append(true[0])
        
        preds = np.array(preds)
        trues = np.array(trues)
        index = np.array(range(1, preds.shape[0] + 1))

        dfData = {
        'index':index,
        'preds':preds.ravel(),
        'trues':trues.ravel()}
        df = pd.DataFrame(dfData)
        df.to_csv(os.path.join(folder_path, setting + '.csv'),index=False)

        
        # f = open("engine.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('preds:{}, trues:{}'.format(preds.ravel(), trues.ravel()))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        visual(preds, trues, os.path.join(folder_path, self.args.data+'#'+str(self.args.engine_num) + '.pdf'))
        
        return


