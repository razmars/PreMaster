import os
import torch
import numpy as np
import neptune
import time
from  utils.tools                import  EarlyStopping, adjust_learning_rate, visual, test_params_flop
from  utils.metrics               import metric
import matplotlib.pyplot as plt


class Exp_Basic(object):
    def __init__(self, args):
        self.args       = args
        self.device     = self._acquire_device()
        self.model      = self._build_model().to(self.device)
        self.neptuneRun = neptune.init_run(project='razmars/preMaster', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjODBjZDM5NC03YzZmLTRhZDUtYWMwYS1jODA0NGE2YTM0MGIifQ==',
                                           name = "razos")
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
        self.neptuneRun["params/predict length"]                    = args.pred_len
        self.neptuneRun["params/seqencec length"]                   = args.seq_len


    def paint_save_test(self,preds,trues,setting,folder_path):
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
            f.write('\n\n')

        self.neptuneRun["results/mse"] = mse 
        self.neptuneRun["results/mae"] = mae 
        self.neptuneRun["results/rse"] = rse 
        np.save(folder_path + 'pred.npy', preds)


    def visual_test(self,i,batch_x,true,pred,folder_path):
        if i % 20 == 0:
            input = batch_x.detach().cpu().numpy()
            gt    = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            pd    = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            self.visual(i,gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        

    def visual(self,i,true, preds=None, name='./pic/test.pdf'):

        for pred in preds:
            self.neptuneRun[f'Charts/predictions-{i}'].append(pred)

        for val in true:
            self.neptuneRun[f'Charts/TrueValue-{i}'].append(val)



    def print_update_inside_epochs(self,i,epoch,train_epochs,train_steps,loss,time_now,iter_count):
        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed      = (time.time() - time_now) / iter_count
            left_time  = speed * ((train_epochs - epoch) * train_steps - i)
            iter_count = 0
            time_now   = time.time()
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

    def updateNeptuneTrainLoss(self,loss):
        self.neptuneRun["Charts/Train Loss"].append(loss)

    def updateNeptuneTestLoss(self,loss):
        self.neptuneRun["Charts/Test Loss"].append(loss)

