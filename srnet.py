import os
import numpy as np
import joblib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable, grad
import wandb

from collections.abc import Iterable
from sjnn import SparseJacobianNN as SJNN
from ghost import GhostAdam, GhostWrapper
from sdnet import SDNet, SDData
import srnet_utils as ut

try:
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        from tqdm.notebook import trange
    else:
        raise RuntimeWarning()
except:
    from tqdm import trange


class SRNet(nn.Module):

    def __init__(self, in_size, out_size, hid_num=1, hid_size=100, hid_type="MLP", hid_kwargs=None, lat_size=25):
        super().__init__()

        # read number, size and type of hidden layers
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

        if isinstance(hid_type, str):
            hid_type1 = hid_type
            hid_type2 = hid_type
        else:
            hid_type1, hid_type2 = hid_type

        if hid_kwargs is None:
            hid_kwargs1 = dict()
            hid_kwargs2 = dict()
        elif isinstance(hid_kwargs, dict):
            hid_kwargs1 = hid_kwargs
            hid_kwargs2 = hid_kwargs
        else:
            hid_kwargs1, hid_kwargs2 = hid_kwargs

        # layers from input to latent
        if hid_type1 == "MLP":
            if hid_num1 == 0:
                layers1 = [nn.Linear(in_size, lat_size)]
            else:
                layers1 = [nn.Linear(in_size, hid_size1), nn.ReLU()]

                for _ in range(hid_num1 - 1):
                    layers1.append(nn.Linear(hid_size1, hid_size1))
                    layers1.append(nn.ReLU())

                layers1.append(nn.Linear(hid_size1, lat_size))

            self.layers1 = nn.Sequential(*layers1)
        
        elif hid_type1 == "DSN":
            self.layers1 = SJNN(in_size, lat_size, hid_size1, hid_num1-1, **hid_kwargs1)

        # layers from latent to output
        if hid_type2 == "MLP":
            if hid_num2 == 0:
                layers2 = [nn.Linear(lat_size, out_size)]
            else:
                layers2 = [nn.Linear(lat_size, hid_size2), nn.ReLU()]

                for _ in range(hid_num2 - 1):
                    layers2.append(nn.Linear(hid_size2, hid_size2))
                    layers2.append(nn.ReLU())

                layers2.append(nn.Linear(hid_size2, out_size))

            self.layers2 = nn.Sequential(*layers2)

        elif hid_type2 == "DSN":
            self.layers2 = SJNN(in_size, lat_size, hid_size2, hid_num2-1, **hid_kwargs2)

    def forward(self, in_data, get_lat=False):

        # propagate from input to latent
        x = self.layers1(in_data)
        lat_acts = x

        # propagate from latent to output
        x = self.layers2(x)

        if get_lat:
            return (x, lat_acts)
        else:
            return x

    def jacobian(self, in_data, get_lat=True):

        # option 1: slow
        # jac = torch.autograd.functional.jacobian(model.layers1, in_data)
        # jac = torch.diagonal(jac, dim1=0, dim2=2).permute(2,0,1)
    
        # option 2: not working
        # jac = functorch.vmap(functorch.jacrev(model.layers1))(in_data)

        # option 3:
        in_data = Variable(in_data, requires_grad=True)
        preds, lat_acts = self(in_data, get_lat=True)

        if get_lat:
            out_data = lat_acts
        else:
            out_data = preds

        grads = []
        for i in range(out_data.shape[1]):
            grad_outputs = torch.zeros_like(out_data)
            grad_outputs[:,i] = 1

            grad_data = grad(
                inputs=in_data,
                outputs=out_data,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]

            grads.append(grad_data)

        return torch.stack(grads).transpose(0,1)


class SRData(Dataset):

    def __init__(self, data_path, in_var=None, lat_var=None, target_var=None, data_mask=None, data_ext=".gz", device=torch.device("cpu")):
        super().__init__()

        # store args
        self.path = data_path
        self.mask = data_mask
        self.ext = data_ext
        self.device = device

        self.in_var = in_var                                          
        self.lat_var = lat_var
        self.target_var = target_var

        # load input data
        if self.in_var:
            self.in_data = self.load_data(self.in_var)
        else:
            self.in_data = None

        # load latent data
        if self.lat_var:
            self.lat_data = self.load_data(self.lat_var)
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


def run_training(model_cls, hyperparams, train_data, val_data=None, disc_cls=None, disc_data=None, load_file=None, disc_file=None, save_file=None, log_freq=1, device=torch.device("cpu"), wandb_project=None):

    # load state for restart
    if load_file:
        state = joblib.load(load_file)
        hyperparams = dict(state['hyperparams'], **hyperparams)

    # load state of discriminator
    if disc_file:
        disc_state = joblib.load(disc_file)
        hyperparams['disc'] = disc_state['hyperparams']['disc']
        try:
            hyperparams['ext'] = disc_state['hyperparams']['ext']
            hyperparams['ext_type'] = disc_state['hyperparams']['ext_type']
            hyperparams['ext_size'] = disc_state['hyperparams']['ext_size']
        except:
            hyperparams['ext'] = None
            hyperparams['ext_type'] = None
            hyperparams['ext_size'] = 0

        # loaded discriminator is not trained further
        hyperparams['disc']['iters'] = 0

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

    # create model
    model = model_cls(**hp['arch']).to(device)
    model.train()
    print(model)

    # create ghost model
    if 'gc' in hp and hp['gc'] > 0:
        model = GhostWrapper(model)

    # create data loader
    loader = DataLoader(train_data, batch_size=hp['batch_size'], shuffle=hp['shuffle'])

    # cost function
    loss_fun = nn.MSELoss()

    # optimizer
    if 'gc' in hp and hp['gc'] > 0:
        optimizer = GhostAdam(model, lr=hp['lr'], weight_decay=hp['wd'], ghost_coeff=hp['gc'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])

    # create discriminator and prediction function
    if 'sd' in hp and hp['sd'] > 0:

        disc_in_size = hp['batch_size']
        try:
            if hp['ext_type'] == "stack":
                disc_in_size *= hp['ext_size'] + 1
            elif hp['ext_type'] == "embed":
                hp['disc']['emb_size'] = hp['ext_size'] + 1
        except: pass

        critic = disc_cls(disc_in_size, **hp['disc']).to(device)
        critic.train()
        print(critic)

        if hp['sd_fun'] == "sigmoid":
            predict = nn.Sigmoid()
        else:
            predict = nn.Identity()

    # monitor statistics
    train_loss = []
    val_loss = []
    corr_mat = []
    disc_preds = []
    stime = time.time()
    times = []
    epoch = 0
    
    # restart discriminator
    if disc_file and 'sd' in hp and hp['sd'] > 0:
        critic.load_state_dict(disc_state['disc_state'])
        critic.optimizer.load_state_dict(disc_state['disc_opt_state'])

    # restart training                                                                              # TODO: check out how to continue a run in wandb
    if load_file:
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        
        if not disc_file and 'sd' in hp and hp['sd'] > 0:
            try:
                critic.load_state_dict(state['disc_state'])
                critic.optimizer.load_state_dict(state['disc_opt_state'])
            except:
                print("Warning: Loading critic parameters failed.")

        train_loss = state['train_loss']
        val_loss = state['val_loss']
        times = state['times']
        stime = time.time() - times[-1]
        epoch = len(train_loss)

        if 'corr_mat' in state:
            corr_mat = state['corr_mat']

        if 'disc_preds' in state:
            disc_preds = state['disc_preds']

        if 'seed_state' in state:
            torch.set_rng_state(state['seed_state'])
    
    # NOTE: watching gradients and parameters is too slow
    # if wandb_project:
    #     batch_num = int(np.ceil(len(train_data) / hp["batch_size"]))
    #     wandb.watch(model, loss_fun, log="all", log_freq=batch_num)

    # define training loop
    t = trange(epoch, hp['epochs'], desc="Epoch")
    for epoch in t:

        batch_loss = []
        batch_disc_preds = []
        for in_data, target_data in loader:

            optimizer.zero_grad()

            preds, lat_acts = model(in_data, get_lat=True)

            loss = loss_fun(preds, target_data)

            if 'gc' in hp and hp['gc'] > 0:
                reg_model = model.ghost
                preds, lat_acts = reg_model(in_data, get_lat=True)
            else:
                reg_model = model

            if 'l1' in hp and hp['l1'] > 0:              
                loss += hp['l1'] * torch.sum(torch.abs(lat_acts)) / lat_acts.shape[1]

            if ('a1' in hp and 'a2' in hp) and (hp['a1'] > 0 or hp['a2'] > 0):
                loss += reg_model.layers1.sparsifying_loss(hp['a1'], hp['a2'])                      # TODO: deal with layers2

            if 'e1' in hp and hp['e1'] > 0:
                loss += reg_model.layers1.entropy_loss(hp['e1'])

            if 'e2' in hp and hp['e2'] > 0:                    
                entropy = reg_model.layers1.entropy()
                var_entropy = F.softmax(lat_acts.var(dim=0)) * entropy
                loss += hp['e2'] * var_entropy.pow(2).sum()

            if 'e3' in hp and hp['e3'] > 0:
                try:
                    data_real = disc_data.get().squeeze(0)
                except:
                    data_real = disc_data.get(in_data=in_data).squeeze(0)

                # v1
                # eps = 1e-6
                # temp = 0.05
                # err = (data_real.unsqueeze(-1) - lat_acts.unsqueeze(0))
                # norm = F.softmax(-1/2 * temp * err.pow(2).mean(dim=1), dim=0) + eps
                # entropy = -(norm * norm.log2()).sum(dim=0)

                # v2
                # eps = 1e-6
                # temp = 0.05
                # var = data_real.unsqueeze(-1).abs() + eps
                # err = (data_real.unsqueeze(-1) - lat_acts.unsqueeze(0)) / var
                # det = var.log().sum(dim=1)
                # norm = F.softmax(-1/2 * temp * (det + err.pow(2).mean(dim=1)), dim=0) + eps
                # entropy = -(norm * norm.log2()).sum(dim=0)

                # v3
                temp = 0.05
                lat_size = lat_acts.shape[1]
                lib_size = data_real.shape[0]
                corr = torch.corrcoef(torch.vstack((lat_acts.T, data_real)))[:lat_size, -lib_size:]
                norm = F.softmax(temp * corr.abs(), dim=1)
                entropy = -(norm * norm.log2()).sum(dim=1)
             
                loss += hp['e3'] * entropy.pow(2).sum()

            if 'sd' in hp and hp['sd'] > 0:

                # get fake extension data
                ext_data_fake = []
                if 'ext' in hp and hp['ext'] is not None:
                    for ext_name in hp['ext']:
                        if ext_name == "input":
                            ext_data_fake.append(in_data)
                        elif ext_name == "grad":
                            grad_data_fake = reg_model.jacobian(in_data, get_lat=True).transpose(0,1)               # lat x batch x grad
                            ext_data_fake.append(grad_data_fake.detach())
                        else:
                            raise KeyError(f"Extension {ext_name} is not defined.")

                if critic.iters > 0:

                    # get fake data
                    data_fake = lat_acts.detach().T
                   
                    # get real data
                    try:
                        datasets_real = disc_data.get(lat_acts.shape[1])
                    except:
                        if disc_data.iter_sample:
                            datasets_real = disc_data.get(lat_acts.shape[1], in_data, critic.iters)
                        else:
                            datasets_real = disc_data.get(lat_acts.shape[1], in_data)
                    
                    dataset_real = datasets_real[...,0]
                
                    # get real extension data
                    ext_data_real = []
                    if 'ext' in hp and hp['ext'] is not None:
                        for ext_name in hp['ext']:
                            if ext_name == "input":
                                ext_data_real.append(in_data)
                            elif ext_name == "grad":
                                grad_data_real = datasets_real[...,1:]                                              # iters x lat x batch x grad
                                ext_data_real.append(grad_data_real)
                            else:
                                raise KeyError(f"Extension {ext_name} is not defined.")
                        
                        # extend real and fake data
                        dataset_real = ut.extend(dataset_real, *ext_data_real, ext_type=hp['ext_type'])
                        data_fake = ut.extend(data_fake, *ext_data_fake, ext_type=hp['ext_type'])
                
                    # train discriminator
                    critic.fit(dataset_real, data_fake)
                
                # extend latent activations
                # TODO: what is the extension data here? do we use grad_data_fake detached or not detached? check softplus activation function
                data_acts = lat_acts.T
                if 'ext' in hp and hp['ext'] is not None:
                    data_acts = ut.extend(data_acts, *ext_data_fake, ext_type=hp['ext_type'])

                # regularize with critic prediction
                p = critic(data_acts)
                loss += hp['sd'] * predict(-p).mean()

                batch_disc_preds.append(p.detach().mean().item())

            loss.backward()

            optimizer.step()

            batch_loss.append(loss.item())

        train_loss.append(np.mean(batch_loss))
        if batch_disc_preds:
            disc_preds.append(np.mean(batch_disc_preds))
        times.append(time.time() - stime)

        if epoch % log_freq == 0 or epoch == hp['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                preds, lat_acts = model(val_data.in_data, get_lat=True)
                val_loss.append(loss_fun(preds, val_data.target_data).item())
                
                if val_data.lat_data is not None:
                    ls = lat_acts.shape[1]
                    corr = torch.corrcoef(torch.hstack((lat_acts, val_data.lat_data)).T)
                    corr_mat.append(corr[:ls, -ls:])

            model.train()
        
            t_update = {"train_loss": train_loss[-1], "val_loss": val_loss[-1]}
            if val_data.lat_data is not None:
                t_update["min_corr"] = corr_mat[-1].abs().max(dim=1).values.min()

            t.set_postfix({k: f"{v:.2e}" if v < 0.1 else f"{v:.2f}" for k, v in t_update.items()})
            if wandb_project:
                t_update["epoch"] = epoch
                t_update["time"] = times[-1]
                if disc_preds:
                    t_update["disc_preds"] = disc_preds[-1]
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

    state = {
        "seed_state": torch.get_rng_state(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "hyperparams": hp._items if wandb_project else hp,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "corr_mat": corr_mat,
        "disc_preds": disc_preds,
        "total_train_loss": total_train_loss,
        "total_val_loss": total_val_loss,
        "times": times,
        "data_path": train_data.path,
        "in_var": train_data.in_var,
        "target_var": train_data.target_var,
        }

    if 'sd' in hp and hp['sd'] > 0:
        state_update = {
            "disc_state": critic.state_dict(),
            "disc_opt_state": critic.optimizer.state_dict(),
            "fun_path": disc_data.path,
            "disc_shuffle": disc_data.shuffle,
            "disc_iter_sample": disc_data.iter_sample,
        }
        state.update(state_update)

    if save_file:
        save_file = save_file.format(**hp, **hp['arch'])
        os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
        joblib.dump(state, save_file)                                                               # TODO: consider device when saving?
        
        if wandb_project:
            wandb.save(save_file)

    if wandb_project:
        wandb.finish()

    return model


if __name__ == '__main__':

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set wandb project
    wandb_project = None # "first-try"
    sweep_id = None
    sweep_num = None

    # select generator and discriminator
    model_cls = SRNet
    disc_cls = SDNet

    # load data
    data_path = "data_1k"

    in_var = "X11"
    lat_var = "G11"
    target_var = "F11"

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

    # create discriminator data
    fun_path = "funs/F11_v1.lib"
    shuffle = True
    iter_sample = False

    if fun_path:
        disc_data = SDData(fun_path, in_var, shuffle=shuffle, iter_sample=iter_sample)
    else:
        disc_data = None
    
    # set load and save file
    load_file = None
    disc_file = None
    save_file = None
    log_freq = 1

    # define hyperparameters
    hyperparams = {
        "arch": {
            "in_size": train_data.in_data.shape[1],
            "out_size": train_data.target_data.shape[1],
            "hid_num": (2, 2),
            "hid_size": (32, 32),
            "hid_type": ("MLP", "MLP"),
            "hid_kwargs": {
                "alpha": None,
                "norm": None,
                "prune": None,
                },
            "lat_size": 3,
            },
        "epochs": 1000,
        "runtime": None,
        "batch_size": train_data.in_data.shape[0],
        "shuffle": False,
        "lr": 1e-4,
        "wd": 1e-7,
        "l1": 0.0,
        "a1": 0.0,
        "a2": 0.0,
        "e1": 0.0,
        "e2": 0.0,
        "e3": 0.0,
        "gc": 0.0,
        "sd": 1e-4,
        "sd_fun": "sigmoid",
        "ext": None,
        "ext_type": None,
        "ext_size": 0,
        "disc": {
            "hid_num": 2,
            "hid_size": 64,
            "lr": 1e-4,
            "wd": 1e-7,
            "betas": (0.9,0.999),
            "iters": 5,
            "gp": 0.0,
            "loss_fun": "BCE",
        },
    }

    run_training(model_cls, hyperparams, train_data, val_data, disc_cls, disc_data, load_file=load_file, disc_file=disc_file, save_file=save_file, log_freq=log_freq, device=device, wandb_project=wandb_project)