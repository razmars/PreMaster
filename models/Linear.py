import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):

        super(Model, self).__init__()
        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear   = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        return self.Linear(x.permute(0,2,1)).permute(0,2,1)
     # [Batch, Output length, Channel]