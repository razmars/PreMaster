import os
import torch
import numpy as np
import pandas as pd
import neptune
import time
from  utils.tools     import  EarlyStopping, adjust_learning_rate, visual, test_params_flop
from  utils.metrics   import metric
from neptune_pytorch  import NeptuneLogger
from neptune.types    import File
from openpyxl         import load_workbook
import matplotlib.pyplot as plt


DATAROW     = {'Electricity':3,'Exchange':7,'traffic':11,'weather':15,'national':19,'ETTh1':23,'ETTh2':27,'ETTm1':31,'ETTm2':35}
STEPROW     = {96:0,192:1,336:2,720:3,24:0,36:1,48:2,60:3}
MODELCOLUMN = {'Linear-MSE':'C','Linear-MAE':'D','NLinear-MSE':'E','NLinear-MAE':'F','DLinear-MSE':'G','DLinear-MAE':'H'}



class Exp_Basic(object):
    def __init__(self, args):
        self.args       = args
        self.device     = self._acquire_device()
        self.model      = self._build_model().to(self.device)
        self.neptuneRun = neptune.init_run(project='razmars/preMaster', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjODBjZDM5NC03YzZmLTRhZDUtYWMwYS1jODA0NGE2YTM0MGIifQ==',
                                           capture_hardware_metrics= True)
        self.neptuneLog = NeptuneLogger(self.neptuneRun,model =self.model, log_model_diagram= True,log_gradients=True,log_parameters = True)
        self.setNeptune(args)
    
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        pass

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


    def setNeptune(self,args):
        self.neptuneRun['model']                             = args.model
        self.neptuneRun['name']                              = args.model_id
        self.neptuneRun["params/batch size"]                 = args.batch_size
        self.neptuneRun["params/learning rate"]              = args.learning_rate
        self.neptuneRun["params/opt algo"]                   = "Adam"
        self.neptuneRun["params/activation"]                 = args.activation
        self.neptuneRun["params/predict length"]             = args.pred_len
        self.neptuneRun["params/seqencec length"]            = args.seq_len
        self.neptuneRun["params/data"]                       = args.model_id.split('_')[0]
        self.neptuneRun["data/raw"].upload(os.path.join(args.root_path,args.data_path))


    def paint_save_test(self,preds,trues,setting,folder_path):
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        self.updateResultsCSV(mse,mae)
        self.neptuneRun["results/mse"] = mse 
        self.neptuneRun["results/mae"] = mae 
        np.save(folder_path + 'pred.npy', preds)

    def updateResultsCSV(self,mse,mae):
        data                  = load_workbook(filename="results.xlsx")
        sheet                 = data.active
        row                   = DATAROW[self.args.model_id.split('_')[0]] + STEPROW[self.args.pred_len]
        colMSE                = MODELCOLUMN[self.args.model+'-MSE']
        colMAE                = MODELCOLUMN[self.args.model+'-MAE']
        celMSE                = str(colMSE)+str(row)
        celMAE                = str(colMAE)+str(row)
        sheet[celMSE]         = mse
        sheet[celMAE]         = mae
        data.save(filename="results.xlsx")
        data = pd.read_excel("results.xlsx")
        data.to_csv("results.csv", index=False)
        sample_df = pd.read_csv("results.csv")
        self.neptuneRun["data/results"].upload(File.as_html(sample_df))
    

    def visual_test(self,i,batch_x,true,pred,folder_path,visualI):
        if i % 20 == 0:
            input = batch_x.detach().cpu().numpy()
            gt    = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            pd    = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            self.visual(i,gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    def visual_test2(self,i,batch_x,true,pred,folder_path,visualI):
        if i == visualI:
            print(visualI)
            print("bitchos")
            print(true[0][0].size)
            for k in range(true[0][0].size):
                input = batch_x.detach().cpu().numpy()
                gt    = np.concatenate((input[0, :, -1], true[0, :, k]), axis=0)
                pd    = np.concatenate((input[0, :, -1], pred[0, :, k]), axis=0)
                self.visual(k,gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        

    def visual(self,i,true, preds=None, name='./pic/test.pdf'):

        for pred in preds:
            self.neptuneRun[f'Charts/predictions-{i}'].append(pred)

        for val in true:
            self.neptuneRun[f'Charts/GroundTrue-{i}'].append(val)



    def print_update_inside_epochs(self,i,epoch,train_epochs,train_steps,loss,time_now,iter_count):
        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed      = (time.time() - time_now) / iter_count
            left_time  = speed * ((train_epochs - epoch) * train_steps - i)
            iter_count = 0
            time_now   = time.time()
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

    def updateNeptuneTrainLoss(self,loss):
        self.neptuneRun[self.neptuneLog.base_namespace]["batch/loss"].append(loss)
       

    def updateNeptuneTestLoss(self,loss):
        self.neptuneRun["Charts/Test Loss"].append(loss)



