import os
import numpy as np
import joblib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable, grad
import wandb

from collections.abc import Iterable
from sdnet import SDNet
import srnet_utils as ut

try:
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        from tqdm.notebook import trange
    else:
        raise RuntimeWarning()
except:
    from tqdm import trange

class CSDNet(nn.Module):

    def __init__(self, in_size, conv_arch=[16], hid_num=1, hid_size=100, emb_size=1, lr=1e-4, wd=1e-7, betas=(0.9,0.999), iters=5, gp=1e-3):
        super().__init__()

        # convolutional layers
        in_channels = emb_size
        layers1 = []

        for a in conv_arch:
            if type(a) == int:
                out_channels = a
                layers1.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='replicate', bias=True))
                layers1.append(nn.ReLU())
                in_channels = a
            else:
                layers1.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
                in_size = -1 * (-in_size // 2)

        self.layers1 = nn.Sequential(*layers1)

        # fully-connected layers
        in_size *= out_channels
        
        if not hid_num:
            layers2 = [nn.Linear(in_size, 1)]
        else:
            layers2 = [nn.Linear(in_size, hid_size), nn.ReLU()]

            for _ in range(hid_num - 1):
                layers2.append(nn.Linear(hid_size, hid_size))
                layers2.append(nn.ReLU())

            layers2.append(nn.Linear(hid_size, 1))

        self.layers2 = nn.Sequential(*layers2)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd, betas=betas)
        self.iters = iters
        self.gp = gp

    def forward(self, x):
        
        # convolution
        x = self.layers1(x)

        # flatten
        x = x.reshape(x.shape[0], -1)

        return self.layers2(x)

    def loss(self, x):
        return self.forward(x).mean()

    def gradient_penalty(self, data_real, data_fake):
        
        try:
            eps = torch.rand_like(data_fake[:,:1,:1])
            grad_dim = (1,2)
        except:
            eps = torch.rand_like(data_fake[:,:1])
            grad_dim = 1

        lat_size = data_fake.shape[0]
        interp = eps * data_real[:lat_size] + (1 - eps) * data_fake
        interp = Variable(interp, requires_grad=True)

        pred = self.forward(interp)

        gradient = grad(
            inputs=interp,
            outputs=pred,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return (gradient.norm(2, dim=grad_dim) - 1).pow(2).mean()

    def fit(self, dataset_real, data_fake):

        data_fake = data_fake.transpose(1,2)
      
        accs = []
        for i in range(self.iters):

            if dataset_real.shape[0] == 1:
                data_real = dataset_real[0].transpose(1,2)
            else:
                data_real = dataset_real[i].transpose(1,2)           

            self.optimizer.zero_grad()

            pred_real = self.forward(data_real)
            pred_fake = self.forward(data_fake)
            
            loss_real = -pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = loss_real + loss_fake
            
            if self.gp:
                loss += self.gp * self.gradient_penalty(data_real, data_fake)

            loss.backward()

            self.optimizer.step()

            with torch.no_grad():
                pred_corr = (pred_real > 0).sum() + (pred_fake <= 0).sum()
                pred_all = pred_real.shape[0] + pred_fake.shape[0]
                accs.append((pred_corr / pred_all).item())

        return accs