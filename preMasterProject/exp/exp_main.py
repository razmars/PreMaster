from data_provider.data_factory import data_provider
from exp.exp_basic              import Exp_Basic
from models                     import  DLinear, Linear, NLinear
from utils.tools                import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics              import metric

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
        pass



    def test(self, setting, test=0):
        pass

    def predict(self, setting, load=False):
        pass