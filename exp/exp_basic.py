import os
import torch
import numpy as np
import neptune


class Exp_Basic(object):
    def __init__(self, args):
        self.args       = args
        self.device     = self._acquire_device()
        self.model      = self._build_model().to(self.device)
        #self.neptuneRun = neptune.init_run(project='razmars/preMaster', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjODBjZDM5NC03YzZmLTRhZDUtYWMwYS1jODA0NGE2YTM0MGIifQ==',
                                           #name = "razos")
        #self.setNeptune(args)
    
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
        self.neptuneRun['namespace/field_name']              = args.model_id
        self.neptuneRun["params/optimization/learning_rate"] = args.learning_rate
        self.neptuneRun["params/optimization/algorithm"]     = "Adam"
        self.neptuneRun["params/activation"]                 = args.activation
        self.neptuneRun["predict length"]                    = args.pred_len
        self.neptuneRun["seqencec length"]                   = args.seq_len