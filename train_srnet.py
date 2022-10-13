import os
import torch
import joblib
from srnet import SRNet, SRData, run_training
from sdnet import SDNet, SDData
import wandb

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set wandb options
wandb_project = "184-DSN-SD-comb-study-F07_v2"
sweep_id = "y6np63gb"
sweep_num = 20

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

# create discriminator data
fun_path = "funs/F07_v2.lib"
shuffle = True
iter_sample = False
    
if fun_path:
    disc_data = SDData(fun_path, in_var, shuffle=shuffle, iter_sample=iter_sample)
else:
    disc_data = None

# set load and save file
load_file = None
disc_file = None
save_file = "models/srnet_model_F07_v2_DSN_SD_comb_study.pkl"
rec_file = None
log_freq = 25

# define hyperparameters
hyperparams = {
    "arch": {
        "in_size": train_data.in_data.shape[1],
        "out_size": train_data.target_data.shape[1],
        "hid_num": (2, 0),
        "hid_size": 32,
        "hid_type": ("DSN", "MLP"),
        "hid_kwargs": {
            "alpha": [[1,0],[0,1],[1,1]],
            "norm": None,
            "prune": None,
            },
        "lat_size": 3,
        },
    "epochs": 50000,
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
    "sd": 1.0,
    "sd_fun": "linear",
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

def train():
    run_training(model_cls, hyperparams, train_data, val_data, disc_cls, disc_data, load_file=load_file, disc_file=disc_file, save_file=save_file, rec_file=rec_file, log_freq=log_freq, device=device, wandb_project=wandb_project)

if __name__ == "__main__":

    # hyperparameter study
    if sweep_id:
        wandb.agent(sweep_id, train, count=sweep_num, project=wandb_project)

    # one training run
    else:
        train()