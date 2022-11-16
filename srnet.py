import os
import numpy as np
import joblib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import wandb

from collections.abc import Iterable
from sdnet import SDNet, SDData
from sjnn import SparseJacobianNN as SJNN

try:
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        from tqdm.notebook import trange
    else:
        raise RuntimeWarning()
except:
    from tqdm import trange


class LinearTransform(nn.Module):

    def __init__(self, in_size, weight=None, bias=None):
        super().__init__()

        if weight is None:
            self.w = nn.Parameter(torch.ones((1, in_size)))
        elif weight == -1:
            self.w = nn.Parameter(torch.ones((1, in_size), requires_grad=False))
        else:
            self.w = nn.Parameter(torch.Tensor(weight).reshape(1,-1), requires_grad=False)

        if bias is None:
            self.b = nn.Parameter(torch.zeros((1, in_size)))
        elif bias == -1:
            self.b = nn.Parameter(torch.zeros((1, in_size)), requires_grad=False)
        else:
            self.b = nn.Parameter(torch.Tensor(bias).reshape(1,-1), requires_grad=False)

    def forward(self, x):
        return x * self.w + self.b


class MLP(nn.Module):

    def __init__(self, in_size, out_size, hid_num, hid_size, **_):
        super().__init__()

        if hid_num == 0:
            layers = [nn.Linear(in_size, out_size)]
        else:
            layers = [nn.Linear(in_size, hid_size), nn.ReLU()]

            for _ in range(hid_num - 1):
                layers.append(nn.Linear(hid_size, hid_size))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hid_size, out_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, get_io_data=False):
        if get_io_data:
            return (self.layers(x), [])
        else:
            return self.layers(x)

    def __getitem__(self, idx):
        return self.layers[idx]


class DSN(nn.Module):

    def __init__(self, in_size, out_size, hid_num, hid_size, alpha=None, norm=None, prune=None, lin_trans=False, **_):
        super().__init__()

        # MLPs
        self.mlps = nn.ModuleList([MLP(in_size, 1, hid_num, hid_size) for _ in range(out_size)])
        
        # input mask
        if alpha is None:
            self.alpha = nn.ParameterList([nn.Parameter(torch.randn(1, in_size)) for _ in range(out_size)])
        elif alpha == -1:
            self.alpha = nn.ParameterList([nn.Parameter(torch.ones(1, in_size), requires_grad=False) for _ in range(out_size)])
        else:            
            self.alpha = nn.ParameterList([nn.Parameter(torch.Tensor(alpha[i]).reshape(1,-1), requires_grad=False) for i in range(out_size)])

        if norm == "softmax":
            self.norm = lambda x: F.softmax(x.abs(), dim=1)
        else:
            self.norm = nn.Identity()
        
        self.prune = prune

        # linear transformations
        if not isinstance(lin_trans, Iterable):
            if lin_trans:
                lin_trans = [((None, None), (None, None)) for _ in range(out_size)]
            else:
                lin_trans = [((-1, -1), (-1, -1)) for _ in range(out_size)]
        
        self.lt_in = nn.ModuleList([LinearTransform(in_size, *lin_trans[i][0]) for i in range(out_size)])
        self.lt_out = nn.ModuleList([LinearTransform(1, *lin_trans[i][1]) for i in range(out_size)])
                
    def forward(self, x, get_io_data=False):

        # loop over MLPs
        preds = []
        io_data = []
        for i in range(len(self.mlps)):

            # input transformation
            y = self.lt_in[i](x)

            if get_io_data:
                in_data = y

            # input mask
            mask = self.norm(self.alpha[i])

            if self.prune:
                mask = torch.where(mask < self.prune, torch.zeros_like(mask), mask)

            y = y * mask

            # MLP
            y = self.mlps[i](y)

            if get_io_data:
                io_data.append((in_data, y))

            # output transformation
            y = self.lt_out[i](y)

            preds.append(y)

        preds = torch.cat(preds, dim=1)

        if get_io_data:
            return (preds, io_data)
        else:
            return preds
            

class SRNet(nn.Module):

    def __init__(self, in_size, lat_size, cell_type="MLP", hid_num=1, hid_size=100, cell_kwargs=None, out_fun=None):
        super().__init__()

        # read network specifications
        if not isinstance(lat_size, Iterable):
            lat_size = [lat_size]

        if isinstance(cell_type, str):
            cell_type = [cell_type]

        if not isinstance(hid_num, Iterable):
            hid_num = [hid_num]

        if not isinstance(hid_size, Iterable):
            hid_size = [hid_size]

        if cell_kwargs is None:
            cell_kwargs = [dict()]
        elif isinstance(cell_kwargs, dict):
            cell_kwargs = [cell_kwargs]

        depth = max(len(lat_size), len(cell_type), len(hid_num), len(hid_size), len(cell_kwargs))

        if len(lat_size) == 1: lat_size *= depth
        if len(cell_type) == 1: cell_type *= depth
        if len(hid_num) == 1: hid_num *= depth
        if len(hid_size) == 1: hid_size *= depth
        if len(cell_kwargs) == 1: cell_kwargs *= depth

        # create cells
        sizes = [in_size, *lat_size]
        self.cells = nn.ModuleList([SJNN(sizes[i], sizes[i+1], hid_size[i], hid_num[i]-1, **cell_kwargs[i]) if cell_type[i] == "SJNN" else globals()[cell_type[i]](sizes[i], sizes[i+1], hid_num[i], hid_size[i], **cell_kwargs[i]) for i in range(depth)])

        # final operation
        if out_fun == "sum":
            self.out = lambda x: x.sum(dim=1).unsqueeze(1)
        elif out_fun is not None:
            self.out = out_fun
        else:
            self.out = nn.Identity()

    def forward(self, x, get_io_data=False):

        # loop over cells
        io_data = []
        lat_data = []
        for cell in self.cells:
            if get_io_data:
                if isinstance(cell, SJNN):
                    x = cell(x)
                    lat_data.append(x)
                else:
                    x, cell_io_data = cell(x, get_io_data)
                    io_data.append(cell_io_data)
                    lat_data.append(x)
            else:
                x = cell(x)
        
        # final operation
        x = self.out(x)

        if get_io_data:
            return (x, io_data, lat_data)
        else:
            return x


class SRData(Dataset):

    def __init__(self, data_path, in_var=None, lat_var=None, target_var=None, data_mask=None, data_ext=".gz", device=torch.device("cpu")):
        super().__init__()

        # store args
        self.path = data_path
        self.mask = data_mask
        self.ext = data_ext
        self.device = device

        self.in_var = in_var
        self.target_var = target_var

        if isinstance(lat_var, str):
            self.lat_var = [lat_var]
        else:
            self.lat_var = lat_var

        # load input data
        if self.in_var:
            self.in_data = self.load_data(self.in_var)
        else:
            self.in_data = None

        # load latent data
        if self.lat_var:
            self.lat_data = [self.load_data(var) for var in self.lat_var]
        else:
            self.lat_data = None

        # load target data
        if self.target_var:
            self.target_data = self.load_data(self.target_var)
        else:
            self.target_data = None

    def load_data(self, var):

        data = np.loadtxt(os.path.join(self.path, var + self.ext))

        if len(data.shape) < 2:
            data = data.reshape(-1, 1)

        if self.mask is not None:
            data = data[self.mask]

        return torch.Tensor(data).to(self.device)                                                   # TODO: all data to device or per batch?

    def __len__(self):
        return self.target_data.shape[0]

    def __getitem__(self, idx):
        return self.in_data[idx], self.target_data[idx]


def run_training(model_cls, hyperparams, train_data, val_data=None, seed=None, disc_cls=None, disc_lib=None, load_file=None, disc_file=None, save_file=None, rec_file=None, log_freq=1, wandb_project=None, device=torch.device("cpu")):

    # set seed
    if seed is None:
        seed = 0
    elif save_file:
        save_file = save_file.replace('.', f"_s{seed}.")

    torch.manual_seed(seed)

    # load state for restart
    if load_file:
        state = joblib.load(load_file)
        hyperparams = dict(state['hyperparams'], **hyperparams)

    # load state of discriminator
    if disc_file:
        disc_state = joblib.load(disc_file)
        hyperparams['disc'] = disc_state['hyperparams']['disc']

        # loaded discriminator is not trained further
        hyperparams['disc']['iters'] = 0

    # initialize wandb
    if wandb_project:
        wandb.init(project=wandb_project, config=hyperparams)
        hp = wandb.config

        if save_file:
            save_file = save_file.replace('.', f"_{wandb.run.id}.")
            wandb.run.name = os.path.basename(save_file.format(**hp, **hp['arch'])).split('.')[0]
    else:
        hp = hyperparams

    # create model
    model = model_cls(**hp['arch']).to(device)
    model.train()
    print(model)

    # create data loader
    loader = DataLoader(train_data, batch_size=hp['batch_size'], shuffle=hp['shuffle'])

    # cost function
    loss_fun = nn.MSELoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])

    # create discriminator and prediction function
    if hp.get('sd', 0):

        critic = disc_cls(hp['batch_size'], **hp['disc']).to(device)
        critic.train()
        print(critic)

        if hp['sd_fun'] == "sigmoid":
            predict = lambda x: F.sigmoid(-x)
        elif hp['sd_fun'] == "logsigmoid":
            predict = lambda x: -F.logsigmoid(x)
        else:
            predict = lambda x: -x

    # monitor statistics
    train_loss = []
    val_loss = []
    corr_mat = []
    disc_preds = []
    disc_loss = []
    rec_io_data = []
    stime = time.time()
    times = []
    epoch = 0
    
    # restart discriminator
    if disc_file and hp.get('sd', 0):
        critic.load_state_dict(disc_state['disc_state'])
        critic.optimizer.load_state_dict(disc_state['disc_opt_state'])

    # restart training                                                                              # TODO: check out how to continue a run in wandb
    if load_file:
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        
        if not disc_file and hp.get('sd', 0):
            try:
                critic.load_state_dict(state['disc_state'])
                critic.optimizer.load_state_dict(state['disc_opt_state'])
            except:
                print("Warning: Loading critic parameters failed.")

        train_loss = state['train_loss']
        val_loss = state['val_loss']
        corr_mat = state['corr_mat']
        disc_preds = state['disc_preds']
        disc_loss = state['disc_loss']
        rec_io_data = state['rec_io_data']
        stime = time.time() - times[-1]
        times = state['times']
        epoch = len(train_loss)

        torch.set_rng_state(state['seed_state'])
    
    # define training loop
    t = trange(epoch, hp['epochs'], desc="Epoch")
    for epoch in t:

        batch_loss = []
        batch_disc_preds = []
        batch_disc_loss = []
        for in_data, target_data in loader:

            optimizer.zero_grad()

            preds, io_data, lat_data = model(in_data, get_io_data=True)

            loss = loss_fun(preds, target_data)

            if hp.get('l1', 0):
                for i in range(len(lat_data)):
                    loss += hp['l1'] * (lat_data[i].abs().sum(dim=0).mean())

            if hp.get('e1', 0):
                for cell in model.cells:
                    try:
                        for a in cell.alpha:
                            an = cell.norm(a)
                            e = -(an * an.log2()).sum()
                            loss += hp['e1'] * e.sum().pow(2)
                    except:
                        pass

            if hp.get('sd', 0):

                if critic.iters:

                    # TODO:
                    # all data for critic training should be detached
                    # a hierarchical model of depth > 1 might have different input dimensions
                    # should we regularize the model until we have the same dimensionality for levels in the hierarchy?
                    # should we train a different critic for each level?
                    # let's hardcode using the first level only for now

                    if isinstance(model.cells[0], SJNN):
                        data_fake = lat_data[0].detach().T
                        datasets_real = disc_lib.get(lat_data[0].shape[1], in_data)
                        dataset_real = datasets_real[...,0]
                    else:
                        # get fake data
                        data_fake = torch.cat([d[1].detach() for d in io_data[0]], dim=1).T
                    
                        # get real data
                        dataset_real = torch.cat([disc_lib.get(1, d[0].detach(), max(1, disc_lib.iter_sample*critic.iters))[...,0] for d in io_data[0]], dim=1)

                    # train discriminator
                    critic.fit(dataset_real, data_fake)
                
                # regularize with critic prediction
                if isinstance(model.cells[0], SJNN):
                    p = critic(lat_data[0].T)
                else:
                    p = critic(torch.cat([d[1] for d in io_data[0]], dim=1).T)                                                  # TODO: hardcoded using first level only
                l = hp['sd'] * predict(p).mean()
                loss += l

                batch_disc_preds.append(p.detach().mean().item())
                batch_disc_loss.append(l.item())

            loss.backward()

            optimizer.step()

            batch_loss.append(loss.item())

        train_loss.append(np.mean(batch_loss))
        if batch_disc_loss:
            disc_preds.append(np.mean(batch_disc_preds))
            disc_loss.append(np.mean(batch_disc_loss))
        times.append(time.time() - stime)

        # evaluate model on validation data
        if epoch % log_freq == 0 or epoch == hp['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                preds, io_data, lat_data = model(val_data.in_data, get_io_data=True)
                val_loss.append(loss_fun(preds, val_data.target_data).item())
                
                if val_data.lat_data is not None:
                    corrs = []
                    for i in range(len(val_data.lat_data)):
                        corr = torch.corrcoef(torch.cat((lat_data[i], val_data.lat_data[i]), dim=1).T)
                        ls = val_data.lat_data[i].shape[1]
                        corrs.append(corr[:ls, -ls:])
                    corr_mat.append(corrs)

                if rec_file is not None:
                    rec_io_data.append(lat_data)

            model.train()
        
            t_update = {"train_loss": train_loss[-1], "val_loss": val_loss[-1]}
            if val_data.lat_data is not None:
                t_update["min_corr"] = min([corr.abs().max(dim=1).values.min().item() for corr in corr_mat[-1]])

            t.set_postfix({k: f"{v:.2e}" if v < 0.1 else f"{v:.2f}" for k, v in t_update.items()})
            if wandb_project:
                t_update["epoch"] = epoch
                t_update["time"] = times[-1]
                if disc_loss:
                    t_update["disc_preds"] = disc_preds[-1]
                    t_update["disc_loss"] = disc_loss[-1]
                wandb.log(t_update)
        
        if hp["runtime"]:
            if times[-1] > hp["runtime"]:
                break

    model.eval()
    with torch.no_grad():
        total_train_loss = loss_fun(model(train_data.in_data), train_data.target_data).item()
        total_val_loss = loss_fun(model(val_data.in_data), val_data.target_data).item()
        print(f"Total training loss: {total_train_loss:.3e}")
        print(f"Total validation loss: {total_val_loss:.3e}")
    model.train()

    state = {
        "seed_state": torch.get_rng_state(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "hyperparams": hp._items if wandb_project else hp,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "corr_mat": corr_mat,
        "disc_preds": disc_preds,
        "disc_loss": disc_loss,
        "times": times,
        "total_train_loss": total_train_loss,
        "total_val_loss": total_val_loss,
        "data_path": train_data.path,
        "in_var": train_data.in_var,
        "target_var": train_data.target_var,
        }

    if hp.get('sd', 0):
        state_update = {
            "disc_state": critic.state_dict(),
            "disc_opt_state": critic.optimizer.state_dict(),
            "fun_path": disc_lib.path,
            "disc_shuffle": disc_lib.shuffle,
            "disc_iter_sample": disc_lib.iter_sample,
        }
        state.update(state_update)

    if save_file:
        save_file = save_file.format(**hp, **hp['arch'])
        os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
        joblib.dump(state, save_file)                                                               # TODO: consider device when saving?
        
        if wandb_project:
            wandb.save(save_file)

    if rec_file:
        os.makedirs(os.path.dirname(os.path.abspath(rec_file)), exist_ok=True)
        joblib.dump(rec_io_data, rec_file)

        # TODO: do we want to save recordings to wandb?
        if wandb_project:
            wandb.save(rec_file)

    if wandb_project:
        wandb.finish()

    return model


if __name__ == '__main__':

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set seed
    seed = None

    # set wandb project
    wandb_project = None # "first-try"
    sweep_id = None
    sweep_num = None

    # select generator and discriminator
    model_cls = SRNet
    disc_cls = SDNet

    # load data
    data_path = "data_1k"

    in_var = "X07"
    lat_var = "G07"
    target_var = "F07"

    mask_ext = ".mask"
    try:
        masks = joblib.load(os.path.join(data_path, in_var + mask_ext))
        train_mask = masks['train']
        val_mask = masks['val']
    except:
        train_mask = None
        val_mask = None
        print("Warning: No masks for training and validation loaded.")

    train_data = SRData(data_path, in_var, lat_var, target_var, train_mask, device=device)
    val_data = SRData(data_path, in_var, lat_var, target_var, val_mask, device=device)

    # load discriminator library
    fun_path = "funs/F07_v2.lib"
    shuffle = True
    iter_sample = False

    if fun_path:
        disc_lib = SDData(fun_path, in_var, shuffle=shuffle, iter_sample=iter_sample)
    else:
        disc_lib = None
    
    # set load, record and save files
    load_file = None
    disc_file = None
    save_file = None
    rec_file = None
    log_freq = 25

    # define hyperparameters
    hyperparams = {
        "arch": {
            "in_size": train_data.in_data.shape[1],
            "lat_size": (3, 1),
            "cell_type": ("SJNN", "MLP"),
            "hid_num": (4, 0),
            "hid_size": 32,
            "cell_kwargs": {
                "alpha": [[1,0],[0,1],[1,1]],
                "norm": None,
                "prune": None,
                # "lin_trans": False, # [[[[1,1], [0,0]], [[2.7], [0]]], [[[1,3.0], [0,0]], [[5.0], [0]]], [[[1,1], [0,0]], [[4.5], [0]]]],    # shape: lat_size x 2 (in/out) x 2 (weight/bias) x in_size/1
                },
            "out_fun": None,
            },
        "epochs": 50000,
        "runtime": None,
        "batch_size": train_data.in_data.shape[0],
        "shuffle": False,
        "lr": 1e-3,
        "wd": 1e-7,
        "l1": 0.0,
        "e1": 0.0,
        "sd": 1e-4,
        "sd_fun": "linear",
        "disc": {
            "hid_num": 2,
            "hid_size": 64,
            "lr": 1e-3,
            "iters": 5,
            "wd": 1e-7,
        },
    }

    run_training(model_cls, hyperparams, train_data, val_data, seed, disc_cls, disc_lib, load_file, disc_file, save_file, rec_file, log_freq, wandb_project, device)