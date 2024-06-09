import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):

        super(Model, self).__init__()
        self.seq_len       = configs.seq_len
        self.pred_len      = configs.pred_len
        #self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        self.individual    = configs.individual
        if self.individual:
            self.Linear            = nn.ModuleList()
            self.clustering_labels = configs.clustering_labels
            self.clustering_groups = configs.clustering_groups
            for i in range(self.clustering_labels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear        = nn.Linear(self.seq_len, self.pred_len)


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x        = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.clustering_labels):
                indices_of_i = [index for index, value in enumerate(self.clustering_groups) if value == i]
                output[:,:,indices_of_i] = self.Linear[i](x.permute(0,2,1)[:,indices_of_i,:]).permute(0,2,1)

            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

        x = x + seq_last
        return x # [Batch, Output length, Channel]
    
