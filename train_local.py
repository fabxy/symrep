import os
import torch
import joblib
from srnet import SRNet, SRData, run_training

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set wandb project
wandb_project = "31-check-convergence"

# load data
data_path = "data"

in_var = "X01"
lat_var = None
target_var = "F01"

mask_ext = ".mask"
masks = joblib.load(os.path.join(data_path, in_var + mask_ext))

train_data = SRData(data_path, in_var, lat_var, target_var, masks["train"], device=device)
val_data = SRData(data_path, in_var, lat_var, target_var, masks["val"], device=device)

# set save file
save_file = f"models/srnet_model_{target_var}_conv.pkl"

# define hyperparameters
hyperparams = {
    "arch": {
        "in_size": train_data.in_data.shape[1],
        "out_size": train_data.target_data.shape[1],
        "hid_num": 2,
        "hid_size": 32, 
        "hid_type": "MLP",
        "lat_size": 16,
        },
    "epochs": 10000,
    "runtime": None,
    "batch_size": 64,
    "lr": 1e-4,
    "wd": 1e-4,
    # "l1": 1e-4,
    "shuffle": True,
}

run_training(SRNet, hyperparams, train_data, val_data, save_file=save_file, device=device, wandb_project=wandb_project);