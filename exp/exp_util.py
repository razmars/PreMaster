import time
import os
import numpy as np
from  utils.tools                import  EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics               import metric


def print_update_inside_epochs(i,epoch,train_epochs,train_steps,loss,time_now,iter_count):
    if (i + 1) % 100 == 0:
        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
        speed      = (time.time() - time_now) / iter_count
        left_time  = speed * ((train_epochs - epoch) * train_steps - i)
        iter_count = 0
        time_now   = time.time()
        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))


def visual_test(i,batch_x,true,pred,folder_path):
       if i % 20 == 0:
        input = batch_x.detach().cpu().numpy()
        gt    = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
        pd    = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

def paint_save_test(preds,trues,setting,folder_path):
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))

    f = open("result.txt", 'a')
    f.write(setting + "  \n")
    f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
    f.write('\n')
    f.write('\n')
    f.close()
    np.save(folder_path + 'pred.npy', preds)


        
