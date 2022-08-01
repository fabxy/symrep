import os
import torch
import joblib
from srnet import SRNet, SRData, run_training
from sdnet import SDNet, SDData
import wandb

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set wandb options
wandb_project = "132-bn-mask-DSN-sd-study-F07_v1"
sweep_id = None
sweep_num = None

# load data
data_path = "data_1k"

in_var = "X07"
lat_var = "G07"
target_var = "F07"

mask_ext = ".mask"
masks = joblib.load(os.path.join(data_path, in_var + mask_ext))

train_data = SRData(data_path, in_var, lat_var, target_var, masks["train"], device=device)
val_data = SRData(data_path, in_var, lat_var, target_var, masks["val"], device=device)

# create discriminator data
fun_path = "funs/F07_v1.lib"
shuffle = True

if fun_path:
    disc_data = SDData(fun_path, in_var, shuffle=shuffle)
else:
    disc_data = None

# set load and save file
load_file = None
save_file = "models/srnet_model_F07_v1_bn_mask_sd_study.pkl"
log_freq = 25

# define hyperparameters
hyperparams = {
    "arch": {
        "in_size": train_data.in_data.shape[1],
        "out_size": train_data.target_data.shape[1],
        "hid_num": (2,0),
        "hid_size": 32, 
        "hid_type": ("DSN", "MLP"),
        "hid_kwargs": {
            "alpha": [[1,0],[0,1],[1,1]],
            "norm": None,
            "prune": None,
            },
        "lat_size": 3,
        },
    "epochs": 30000,
    "runtime": None,
    "batch_size": train_data.in_data.shape[0],
    "shuffle": False,
    "lr": 1e-4,
    "wd": 1e-6,
    "l1": 0.0,
    "a1": 0.0,
    "a2": 0.0,
    "e1": 0.0,
    "e2": 0.0,
    "e3": 0.0,
    "gc": 0.0,
    "sd": 1e-6,
    "disc": {
        "hid_num": 6,
        "hid_size": 128,
        "emb_size": None,
        "lr": 1e-3,
        "wd": 1e-4,
        "iters": 5,
        "gp": 1e-5,
    },
}

def train():
    run_training(SRNet, hyperparams, train_data, val_data, SDNet, disc_data, load_file=load_file, save_file=save_file, log_freq=log_freq, device=device, wandb_project=wandb_project)

if __name__ == "__main__":

    # hyperparameter study
    if sweep_id:
        wandb.agent(sweep_id, train, count=sweep_num, project=wandb_project)

    # one training run
    else:
        train()