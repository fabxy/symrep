import os
import numpy as np
import joblib
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import wandb

from collections.abc import Iterable

try:
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        from tqdm.notebook import trange
    else:
        raise RuntimeWarning()
except:
    from tqdm import trange


class SRNet(nn.Module):

    def __init__(self, in_size, out_size, hid_num=1, hid_size=100, hid_type="MLP", lat_size=25):
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

        # layers from input to latent
        if hid_num1 == 0:
            layers1 = [nn.Linear(in_size, lat_size)]
        else:
            layers1 = [nn.Linear(in_size, hid_size1), nn.ReLU()]

            for _ in range(hid_num1 - 1):
                layers1.append(nn.Linear(hid_size1, hid_size1))
                layers1.append(nn.ReLU())

            layers1.append(nn.Linear(hid_size1, lat_size))

        self.layers1 = nn.Sequential(*layers1)

        # layers from latent to output
        if hid_num2 == 0:
            layers2 = [nn.Linear(lat_size, out_size)]
        else:
            layers2 = [nn.Linear(lat_size, hid_size2), nn.ReLU()]

            for _ in range(hid_num2 - 1):
                layers2.append(nn.Linear(hid_size2, hid_size2))
                layers2.append(nn.ReLU())

            layers2.append(nn.Linear(hid_size2, out_size))

        self.layers2 = nn.Sequential(*layers2)

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
            data = data.reshape(-1, 1)                                  # Q: is this necessary?

        if self.mask is not None:
            data = data[self.mask]

        return torch.Tensor(data).to(self.device)                       # Q: all data to device or per batch?

    def __len__(self):
        return self.target_data.shape[0]

    def __getitem__(self, idx):
        return self.in_data[idx], self.target_data[idx]


def run_training(model_cls, hyperparams, train_data, val_data=None, load_file=None, save_file=None, device=torch.device("cpu"), wandb_project=None):
    """
    TODO: 
    - Implement restart option
    """

    # initialize wandb
    if wandb_project:
        wandb.init(project=wandb_project, config=hyperparams)
        hp = wandb.config
        if save_file:
            wandb.run.name = (os.path.basename(save_file).split('.')[0]).format(**hp, **hp['arch'])
    else:
        hp = hyperparams

    # set seed
    torch.manual_seed(0)

    # create model
    model = model_cls(**hp['arch']).to(device)
    model.train()

    # create data loader
    loader = DataLoader(train_data, batch_size=hp['batch_size'], shuffle=hp['shuffle'])

    # cost function
    loss_fun = nn.MSELoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])

    # define training loop
    train_loss = []
    val_loss = []
    stime = time.time()
    times = []

    # if wandb_project:                                                 # NOTE: watching gradients and parameters is too slow
    #     batch_num = int(np.ceil(len(train_data) / hp["batch_size"]))
    #     wandb.watch(model, loss_fun, log="all", log_freq=batch_num)

    t = trange(hp['epochs'], desc="Epoch")
    for epoch in t:

        batch_loss = []
        for in_data, target_data in loader:

            optimizer.zero_grad()

            preds, lat_acts = model(in_data, get_lat=True)

            loss = loss_fun(preds, target_data)

            if 'l1' in hp and hp['l1'] > 0:
                loss += hp['l1'] * torch.sum(torch.abs(lat_acts)) / lat_acts.shape[1]

            loss.backward()

            optimizer.step()

            batch_loss.append(loss.item())

        train_loss.append(np.mean(batch_loss))

        model.eval()
        with torch.no_grad():
            val_loss.append(loss_fun(model(val_data.in_data), val_data.target_data).item())
        model.train()
        times.append(time.time() - stime)

        t.set_postfix({"train_loss": f"{train_loss[-1]:.2e}", "val_loss": f"{val_loss[-1]:.2e}"})
        if wandb_project:
            wandb.log({"epoch": epoch, "time": times[-1], "train_loss": train_loss[-1], "val_loss": val_loss[-1]})
        
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
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "hyperparams": hp._items if wandb_project else hp,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "total_train_loss": total_train_loss,
        "total_val_loss": total_val_loss,
        "times": times,
        "data_path": train_data.path,
        "in_var": train_data.in_var,
        "target_var": train_data.target_var,
        }
    
    if save_file:
        os.makedirs(data_path, exist_ok=True)
        joblib.dump(state, save_file.format(**hp, **hp['arch']))        # TODO: consider device when saving?
        
        if wandb_project:
            wandb.save(save_file)

    if wandb_project:
        wandb.finish()

    return model


if __name__ == '__main__':

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set wandb project
    wandb_project = "first-try"

    # load data
    data_path = "data"
    
    in_var = "X01"
    lat_var = None
    target_var = "F01"

    mask_ext = ".mask"
    masks = joblib.load(os.path.join(data_path, in_var + mask_ext))     # TODO: create mask if file does not exist

    train_data = SRData(data_path, in_var, lat_var, target_var, masks["train"], device=device)
    val_data = SRData(data_path, in_var, lat_var, target_var, masks["val"], device=device)

    # set save file
    save_file = None

    # define hyperparameters
    hyperparams = {
        "arch": {
            "in_size": train_data.in_data.shape[1],
            "out_size": train_data.target_data.shape[1],
            "hid_num": 3,
            "hid_size": (50, 25), 
            "hid_type": "MLP",
            "lat_size": 10,
            },
        "epochs": 1000,
        "runtime": None,
        "batch_size": 50,
        "lr": 1e-4,                                                     # TODO: adaptive learning rate?
        "wd": 1e-4,
        "l1": 0.0,
        "shuffle": True,
    }

    run_training(SRNet, hyperparams, train_data, val_data, save_file=save_file, device=device, wandb_project=wandb_project)