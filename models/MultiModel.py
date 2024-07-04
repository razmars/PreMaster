import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models                     import  DLinear, Linear, NLinear


class Model(nn.Module):
    def __init__(self, configs):

        super(Model, self).__init__()
        model_dict = {'DLinear': DLinear,'NLinear': NLinear,'Linear': Linear}
        #model = model_dict[self.args.model].Model(self.args).float()
        self.seq_len       = configs.seq_len
        self.pred_len      = configs.pred_len
        #self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        self.individual    = configs.individual
        if self.individual:
            configs.individual     = False
            self.models_list       = nn.ModuleList()
            self.clustering_labels = configs.clustering_labels
            self.clustering_groups = configs.clustering_groups
            self.clustering_models = configs.clustering_models
            for model_i in self.clustering_models:
                self.models_list.append(model_dict[model_i].Model(configs).float())
        else:
            self.Linear        = nn.Linear(self.seq_len,self.pred_len)

    
    def forward(self, x):
        if self.individual:
            batch_size, _, num_features = x.size()
            output = torch.zeros([batch_size, self.pred_len, num_features], dtype=x.dtype).to(x.device)
            for i in range(len(self.clustering_groups)):
                # Ensure the input to each model has the correct shape
                model_input = x[:, :, i].unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
                model_output = self.models_list[i](model_input)  # Shape: [batch_size, pred_len, 1]
                output[:, :, i] = model_output.squeeze(-1)  # Remove the last dimension
        else:
            output = self.Linear(x)
        return output
     