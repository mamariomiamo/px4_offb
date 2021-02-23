"""
This file defines the class of neural network that parameterizes Q and R
------------------------------------------------------------------------
Wang, Bingheng, 02, Jan., 2021, at UTown, NUS
modified on 08, Jan., 2021, at Control & Simulation Lab, NUS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        # D_in : dimension of input layer
        # D_h  : dimension of hidden layer
        # D_out: dimension of output layer
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, D_h)
        self.linear2 = nn.Linear(D_h, D_out)
        self.hidden_size = D_h

    def forward(self, input):
        # convert state s to tensor
        S = torch.tensor(input, dtype=torch.float) # column 2D tensor
        z1 = self.linear1(S.t()) # linear function requires the input to be a row tensor
        Alpha = torch.tensor(0.25, dtype=torch.float)
        alpha = nn.Parameter(Alpha)
        z2 = F.prelu(z1, alpha) # hidden layer
        z3 = self.linear2(z2)
        para = torch.sigmoid(z3) # output layer, row 2D tensor
        return para.t()

    def myloss(self, para, dp):
        # convert np.array to tensor
        Dp = torch.tensor(dp, dtype=torch.float) # row 2D tensor
        loss_nn = torch.matmul(Dp, para)
        return loss_nn

class NetCtrl(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        # D_in : dimension of input layer
        # D_h  : dimension of hidden layer
        # D_out: dimension of output layer
        super(NetCtrl, self).__init__()
        self.linear1 = nn.Linear(D_in, D_h)
        self.linear2 = nn.Linear(D_h, D_out)
        self.hidden_size = D_h

    def forward(self, input):
        # convert state s to tensor
        S = torch.tensor(input, dtype=torch.float) # column 2D tensor
        z1 = self.linear1(S.t()) # linear function requires the input to be a row tensor
        Alpha = torch.tensor(0.25, dtype=torch.float)
        alpha = nn.Parameter(Alpha)
        z2 = F.prelu(z1, alpha) # hidden layer
        z3 = self.linear2(z2)
        para_gain = torch.sigmoid(z3) # output layer, row 2D tensor
        return para_gain.t()

    def myloss(self, para_gain, dg):
        # convert np.array to tensor
        Dg = torch.tensor(dg, dtype=torch.float) # row 2D tensor
        loss_nn = torch.matmul(Dg, para_gain)
        return loss_nn




