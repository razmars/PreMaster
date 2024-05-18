import numpy             as np
import pandas            as pd
import torch.nn          as nn
import torch
import matplotlib.pyplot as plt
import neptune
import warnings
import os
import time

from data_provider.data_factory import  data_provider
from exp.exp_basic              import  Exp_Basic
from models                     import  DLinear, Linear, NLinear
from utils.tools                import  EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics              import  metric
from exp.exp_util               import  print_update_inside_epochs,visual_test,paint_save_test
from torch                      import optim

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        

    def _build_model(self):
        model_dict = {'DLinear': DLinear,'NLinear': NLinear,'Linear': Linear}
        model = model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        print("args")
        print(self.args)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        path                     = os.path.join(self.args.checkpoints, setting)
        time_now                 = time.time()
        train_steps              = len(train_loader)
        early_stopping           = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim              = self._select_optimizer()
        criterion                = self._select_criterion()

        if not os.path.exists(path): os.makedirs(path)
                
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                iter_count += 1
                batch_x     = batch_x.float().to(self.device)
                batch_y     = batch_y.float().to(self.device)
                outputs     = self.model(batch_x)

                f_dim       = -1 if self.args.features == 'MS' else 0
                outputs     = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y     = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss        = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                #self.neptuneRun["train_loss"].append(loss.item())
                print_update_inside_epochs(i,epoch,self.args.train_epochs,train_steps,loss,time_now,iter_count)

                loss.backward()
                model_optim.step()


            train_loss = np.average(train_loss)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop: print("Early stopping"); break

            adjust_learning_rate(model_optim, epoch + 1, self.args)


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        preds                  = []
        trues                  = []
        inputx                 = [] 
        folder_path            = './test_results/' + setting + '/'

        if test:print('loading model');self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        if not os.path.exists(folder_path):os.makedirs(folder_path)

        self.model.eval()

        with torch.no_grad():

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred    = outputs
                true    = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                visual_test(i,batch_x,true,pred,folder_path)


        #if self.args.test_flop:test_params_flop((batch_x.shape[1],batch_x.shape[2]));exit()

        preds  = np.concatenate(preds, axis=0)
        trues  = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):os.makedirs(folder_path)
        paint_save_test(preds,trues,setting,folder_path)

        return


    def predict(self, setting, load=False):

        pred_data, pred_loader = self._get_data(flag='pred')
        preds                  = []

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        with torch.no_grad():

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                outputs = self.model(batch_x)

                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        print(f'preds: {preds}')
        print(f'preds_data: {pred_data}')
        #print(f'pred_data size: {pred_data.size()}')
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)

        print(preds)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds, axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return



        

