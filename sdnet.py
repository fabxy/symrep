import os
import numpy as np
import joblib
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable, grad

class SDNet(nn.Module):

    def __init__(self, in_size, hid_num=1, hid_size=100, lr=1e-4, wd=1e-7, iters=5, gp=1e-3):
        super().__init__()

        if not hid_num:
            layers = [nn.Linear(in_size, 1)]
        else:
            layers = [nn.Linear(in_size, hid_size), nn.ReLU()]

            for _ in range(hid_num - 1):
                layers.append(nn.Linear(hid_size, hid_size))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hid_size, 1))

        self.layers = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        self.iters = iters
        self.gp = gp

    def forward(self, x):
        return self.layers(x)

    def loss(self, x):
        return self.forward(x).mean()

    def gradient_penalty(self, data_real, data_fake):
        
        alpha = torch.rand_like(data_real[:,:1])
        interp = alpha * data_real + (1 - alpha) * data_fake
        interp = Variable(interp, requires_grad=True)

        pred = self.forward(interp)

        gradient = grad(
            inputs=interp,
            outputs=pred,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return (gradient.norm(2, dim=1) - 1).pow(2).mean()


    def fit(self, data_real, data_fake):
      
        losses = []
        for _ in range(self.iters):

            self.optimizer.zero_grad()

            loss_real = -self.loss(data_real)
            loss_fake = self.loss(data_fake)
            loss_gp = self.gradient_penalty(data_real, data_fake)

            loss = loss_real + loss_fake + self.gp * loss_gp

            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())

        return losses


class SDData(Dataset):

    def __init__(self, fun_path, in_var, in_data=None):
        super().__init__()

        self.in_var = in_var
        self.path = fun_path

        with open(fun_path, 'r') as f:
            self.funs = [fun.strip() for fun in f]
            self.len = len(self.funs)

        if in_data is not None:
            self.fun_data = self.evaluate(self.funs, in_data)
        else:
            self.fun_data = None

    def evaluate(self, funs, in_data):
        data_dict = {self.in_var: in_data}
        return torch.vstack([eval(fun, {'np': np}, data_dict) for fun in funs])
    
    def get(self, fun_num=None, in_data=None):                 # TODO: get or __get__?

        if fun_num is None:
            fun_num = self.len
        
        idxs = torch.randperm(self.len)[:fun_num]
        
        if in_data is None:
            if self.fun_data is None:
                raise RuntimeError("No input data provided.")
            else:
                return self.fun_data[idxs]
        else:
            funs = [self.funs[idx] for idx in idxs]
            return self.evaluate(funs, in_data)
