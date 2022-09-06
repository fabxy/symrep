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
import srnet_utils as ut

import os
import numpy as np
import joblib
import time

try:
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        from tqdm.notebook import trange
    else:
        raise RuntimeWarning()
except:
    from tqdm import trange


class SDNet(nn.Module):

    def __init__(self, in_size, hid_num=1, hid_size=100, emb_size=None, lr=1e-4, wd=1e-7, betas=(0.9,0.999), iters=5, gp=1e-3):
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
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd, betas=betas)
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


    def fit(self, dataset_real, data_fake):
      
        accs = []
        for i in range(self.iters):

            if dataset_real.shape[0] == 1:
                data_real = dataset_real[0]
            else:
                data_real = dataset_real[i]
                    
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


class SDData(Dataset):

    def __init__(self, fun_path, in_var, in_data=None, shuffle=True, iter_sample=False):
        super().__init__()

        self.in_var = in_var
        self.path = fun_path
        self.shuffle = shuffle
        self.iter_sample = iter_sample

        with open(fun_path, 'r') as f:
            self.funs = [fun.strip().split(';') for fun in f]
            self.len = len(self.funs)

        if in_data is not None:
            self.fun_data = self.evaluate(self.funs, in_data)
        else:
            self.fun_data = None

    def evaluate(self, funs, in_data):
        data_dict = {self.in_var: in_data}
        
        eval_data = []
        for fun in funs:
            coeff_dict = {}
            c = 0
            while f"N{c}" in fun[0] or f"U{c}" in fun[0]:
                coeff_dict[f"N{c}"] = torch.randn(1).item()
                coeff_dict[f"U{c}"] = torch.rand(1).item()
                c += 1
            
            fun_data = [eval(f, {'np': np}, {**data_dict, **coeff_dict}) for f in fun]
            fun_data = [f_data if torch.is_tensor(f_data) else f_data * torch.ones(data_dict[self.in_var].shape[0]) for f_data in fun_data]
            eval_data.append(torch.stack(fun_data).T)

        return torch.stack(eval_data)
    
    def get(self, fun_num=None, in_data=None, iter_num=1):                 # TODO: get or __get__?

        if fun_num is None:
            fun_num = self.len

        iter_data = []
        for _ in range(iter_num):
        
            if self.shuffle:
                # NOTE: when resampling coeffs, we can draw with replacement
                # idxs = torch.randperm(self.len)[:fun_num]
                idxs = torch.randint(self.len, (fun_num,))
            else:
                idxs = torch.arange(fun_num)
        
            if in_data is None:
                if self.fun_data is None:
                    raise RuntimeError("No input data provided.")
                else:
                    iter_data.append(self.fun_data[idxs])
            else:
                funs = [self.funs[idx] for idx in idxs]
                iter_data.append(self.evaluate(funs, in_data))

        return torch.stack(iter_data)


def run_training(disc_cls, model_cls, hyperparams, train_data, disc_data, load_file=None, save_file=None, log_freq=1, acc_hor=100, device=torch.device("cpu"), wandb_project=None):

    # load state for restart
    if load_file:
        state = joblib.load(load_file)
        hyperparams = dict(state['hyperparams'], **hyperparams)

    # initialize wandb
    if wandb_project:
        wandb.init(project=wandb_project, config=hyperparams)
        hp = wandb.config
        if save_file:
            wandb.run.name = os.path.basename(save_file.format(**hp, **hp['arch'])).split('.')[0]
    else:
        hp = hyperparams

    # set seed
    torch.manual_seed(0)

    # create discriminator
    disc_in_size = hp['batch_size']
    try:
        if hp['ext_type'] == "stack":
            disc_in_size *= hp['ext_size'] + 1
        elif hp['ext_type'] == "embed":
            hp['disc']['emb_size'] = hp['ext_size'] + 1
    except: pass

    critic = disc_cls(disc_in_size, **hp['disc']).to(device)
    critic.train()

    # monitor statistics
    tot_accs = []
    stime = time.time()
    times = []
    epoch = 0
    
    # restart training
    if load_file:        
        critic.load_state_dict(state['disc_state'])
        critic.optimizer.load_state_dict(state['disc_opt_state'])

        tot_accs = state['tot_accs']
        times = state['times']
        stime = time.time() - times[-1]
        epoch = len(times)

        torch.set_rng_state(state['seed_state'])
    
    # define training loop
    t = trange(epoch, hp['epochs'], desc="Epoch")
    for epoch in t:

        # get fake data
        model = model_cls(**hp['arch']).to(device)
        model.train()

        with torch.no_grad():
            _, lat_acts = model(train_data.in_data, get_lat=True)

        data_fake = lat_acts.detach().T

        # get real data
        if disc_data.iter_sample:
            datasets_real = disc_data.get(lat_acts.shape[1], train_data.in_data, critic.iters)
        else:
            datasets_real = disc_data.get(lat_acts.shape[1], train_data.in_data)

        dataset_real = datasets_real[...,0]

        # extend real and fake data
        ext_data_real = []
        ext_data_fake = []
        if 'ext' in hp and hp['ext'] is not None:
            for ext_name in hp['ext']:
                if ext_name == "input":
                    ext_data_real.append(train_data.in_data)
                    ext_data_fake.append(train_data.in_data)
                elif ext_name == "grad":
                    grad_data_real = datasets_real[...,1:]
                    grad_data_fake = model.jacobian(train_data.in_data, get_lat=True).transpose(0,1)
                    ext_data_real.append(grad_data_real)
                    ext_data_fake.append(grad_data_fake.detach())
                else:
                    raise KeyError(f"Extension {ext_name} is not defined.")
            
            dataset_real = ut.extend(dataset_real, *ext_data_real, ext_type=hp['ext_type'])
            data_fake = ut.extend(data_fake, *ext_data_fake, ext_type=hp['ext_type'])

        # train critic
        accs = critic.fit(dataset_real, data_fake)

        # monitor stats
        tot_accs.append(np.mean(accs))
        times.append(time.time() - stime)

        if epoch % log_freq == 0 or epoch == hp['epochs'] - 1:
                    
            t_update = {"acc": tot_accs[-1], "avg_acc": np.mean(tot_accs[-acc_hor:])}
            t.set_postfix({k: f"{v:.2f}" for k, v in t_update.items()})
            if wandb_project:
                t_update["epoch"] = epoch
                t_update["time"] = times[-1]
                wandb.log(t_update)
        
        if hp["runtime"]:
            if times[-1] > hp["runtime"]:
                break

    state = {
        "seed_state": torch.get_rng_state(),
        "hyperparams": hp._items if wandb_project else hp,
        "tot_accs": tot_accs,
        "times": times,
        "data_path": train_data.path,
        "in_var": train_data.in_var,
        "disc_state": critic.state_dict(),
        "disc_opt_state": critic.optimizer.state_dict(),
        "fun_path": disc_data.path,
        "disc_shuffle": disc_data.shuffle,
        "disc_iter_sample": disc_data.iter_sample,
        }

    if save_file:
        save_file = save_file.format(**hp, **hp['arch'])
        os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
        joblib.dump(state, save_file)
        
        if wandb_project:
            wandb.save(save_file)

    if wandb_project:
        wandb.finish()

    return critic