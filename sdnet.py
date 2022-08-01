import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable, grad

from collections.abc import Iterable

class SDNet(nn.Module):

    def __init__(self, in_size, hid_num=1, hid_size=100, emb_size=None, lr=1e-4, wd=1e-7, iters=5, gp=1e-3):
        super().__init__()

        self.in_size = in_size

        if not isinstance(hid_num, Iterable):
            hid_num1 = hid_num
            hid_num2 = hid_num
        else:
            hid_num1, hid_num2 = hid_num

        if not isinstance(hid_size, Iterable):
            hid_size1 = hid_size
            hid_size2 = hid_size
        else:
            hid_size1, hid_size2 = hid_size

        if emb_size is None or emb_size <= 1:
            self.emb_size = 1
        else:
            self.emb_size = emb_size

        # embedding layers
        if self.emb_size == 1:
            self.layers1 = nn.Identity()
        else:
            if not hid_num1:
                layers1 = [nn.Linear(self.emb_size, 1)]
            else:
                layers1 = [nn.Linear(self.emb_size, hid_size1), nn.ReLU()]

                for _ in range(hid_num1 - 1):
                    layers1.append(nn.Linear(hid_size1, hid_size1))
                    layers1.append(nn.ReLU())

                layers1.append(nn.Linear(hid_size1, 1))

            self.layers1 = nn.Sequential(*layers1)

        # discriminator layers
        if not hid_num2:
            layers2 = [nn.Linear(self.in_size, 1)]
        else:
            layers2 = [nn.Linear(self.in_size, hid_size2), nn.ReLU()]

            for _ in range(hid_num2 - 1):
                layers2.append(nn.Linear(hid_size2, hid_size2))
                layers2.append(nn.ReLU())

            layers2.append(nn.Linear(hid_size2, 1))

        self.layers2 = nn.Sequential(*layers2)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        self.iters = iters
        self.gp = gp

    def forward(self, x):
        
        # embedding
        if self.emb_size > 1:
            x = x.reshape(-1, self.emb_size)
            x = self.layers1(x)
            x = x.reshape(-1, self.in_size)
        
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


    def fit(self, data_real, data_fake):
      
        losses = []
        for _ in range(self.iters):

            self.optimizer.zero_grad()

            loss_real = -self.loss(data_real)
            loss_fake = self.loss(data_fake)
            loss = loss_real + loss_fake
            
            if self.gp:
                loss += self.gp * self.gradient_penalty(data_real, data_fake)

            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())

        return losses


class SDData(Dataset):

    def __init__(self, fun_path, in_var, in_data=None, shuffle=True):
        super().__init__()

        self.in_var = in_var
        self.path = fun_path
        self.shuffle = shuffle

        with open(fun_path, 'r') as f:
            self.funs = [fun.strip() for fun in f]
            self.len = len(self.funs)

        if in_data is not None:
            self.fun_data = self.evaluate(self.funs, in_data)
        else:
            self.fun_data = None

    def evaluate(self, funs, in_data):
        data_dict = {self.in_var: in_data}
        
        eval_data = []
        for fun in funs:

            while 'N' in fun:
                fun = fun.replace('N', str(torch.randn(1).item()), 1)
                
            while 'U' in fun:
                fun = fun.replace('U', str(torch.rand(1).item()), 1)

            eval_data.append(eval(fun, {'np': np}, data_dict))

        return torch.vstack(eval_data)
    
    def get(self, fun_num=None, in_data=None):                 # TODO: get or __get__?

        if fun_num is None:
            fun_num = self.len
        
        if self.shuffle:
            idxs = torch.randperm(self.len)[:fun_num]
        else:
            idxs = torch.arange(fun_num)
        
        if in_data is None:
            if self.fun_data is None:
                raise RuntimeError("No input data provided.")
            else:
                return self.fun_data[idxs]
        else:
            funs = [self.funs[idx] for idx in idxs]
            return self.evaluate(funs, in_data)
