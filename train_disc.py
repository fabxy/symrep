import os
import torch
import joblib
from srnet import SRNet, SRData
from sdnet import SDNet, SDData, run_training
from csdnet import CSDNet
import wandb

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set wandb options
wandb_project = "162-ext2-study-F11_v1"
sweep_id = "c5hgi21o"
sweep_num = 15

# select generator and discriminator
model_cls = SRNet
disc_cls = SDNet

# load data
data_path = "data_1k"
in_var = "X11"
lat_var = None
target_var = None

# mask_ext = ".mask"
# masks = joblib.load(os.path.join(data_path, in_var + mask_ext))

train_data = SRData(data_path, in_var, data_mask=None, device=device)

# create discriminator data
fun_path = "funs/F11_v1.lib"
shuffle = True
iter_sample = False
disc_data = SDData(fun_path, in_var, shuffle=shuffle, iter_sample=iter_sample)

# set load and save file
load_file = None
save_file = "models/disc_model_F11_v1_ext2_study.pkl"
log_freq = 25
acc_hor = 500

# define hyperparameters
hyperparams = {
    "arch": {
        "in_size": train_data.in_data.shape[1],
        "out_size": 1,
        "hid_num": (2,0),
        "hid_size": 32, 
        "hid_type": ("MLP", "MLP"),
        "hid_kwargs": {
            "alpha": None,
            "norm": None,
            "prune": None,
            },
        "lat_size": 3,
    },
    "epochs": 100000,
    "runtime": None,
    "batch_size": train_data.in_data.shape[0],
    "ext": ["grad"],
    "ext_type": "embed",
    "ext_size": 1,
    "disc": {
        # "conv_arch": [8, 8, 'M', 16, 16, 'M', 32, 'M'],
        # "kernel_size": 3,
        "hid_num": 2,
        "hid_size": 64,
        "lr": 1e-4,
        "wd": 1e-7,
        "betas": (0.9,0.999),
        "iters": 5,
        "gp": 1e-5,
    },
}

def train():
    run_training(disc_cls, model_cls, hyperparams, train_data, disc_data, load_file=load_file, save_file=save_file, log_freq=log_freq, acc_hor=acc_hor, device=device, wandb_project=wandb_project)

if __name__ == "__main__":

    # hyperparameter study
    if sweep_id:
        wandb.agent(sweep_id, train, count=sweep_num, project=wandb_project)

    # one training run
    else:
        train()