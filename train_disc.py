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
wandb_project = None
sweep_id = None
sweep_num = None

# select generator and discriminator
model_cls = SRNet
disc_cls = SDNet

# load data
data_path = "data_1k"
in_var = "X07"
lat_var = None
target_var = None

mask_ext = ".mask"
try:
    masks = joblib.load(os.path.join(data_path, in_var + mask_ext))
    mask = masks['train']
except:
    mask = None
    print("Warning: No mask for training loaded.")

train_data = SRData(data_path, in_var, data_mask=mask, device=device)
# train_data.in_data = train_data.in_data[:,:1].sort(0)[0]

# create discriminator data
fun_path = "funs/F07_v2.lib"
shuffle = True
iter_sample = False
disc_data = SDData(fun_path, in_var, shuffle=shuffle, iter_sample=iter_sample)

# set load and save file
load_file = None
save_file = "models/disc_model_F07_v2_fixed_BCE.pkl"
log_freq = 25
avg_hor = 500

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
    "epochs": 20000,
    "runtime": None,
    "batch_size": train_data.in_data.shape[0],
    # "ext": [],
    # "ext_type": "embed",
    # "ext_size": 0,
    "disc": {
        # "conv_arch": [8, 8, 'M', 16, 16, 'M', 32, 'M'],
        # "kernel_size": 3,
        # "hid_num": 2,
        # "hid_size": 64,
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

def train():
    run_training(disc_cls, model_cls, hyperparams, train_data, disc_data, load_file=load_file, save_file=save_file, log_freq=log_freq, avg_hor=avg_hor, device=device, wandb_project=wandb_project)

if __name__ == "__main__":

    # hyperparameter study
    if sweep_id:
        wandb.agent(sweep_id, train, count=sweep_num, project=wandb_project)

    # one training run
    else:
        train()