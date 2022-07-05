import os
import torch
import joblib
from srnet import SRNet, SRData, run_training
import wandb

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set wandb options
wandb_project = "61-l1-gc-study-F00"
sweep_id = "xnfh3aes"
sweep_num = 9

# load data
data_path = "data_1k"

in_var = "X00"
lat_var = None
target_var = "F00"

mask_ext = ".mask"
masks = joblib.load(os.path.join(data_path, in_var + mask_ext))

train_data = SRData(data_path, in_var, lat_var, target_var, masks["train"], device=device)
val_data = SRData(data_path, in_var, lat_var, target_var, masks["val"], device=device)

# set save file
save_file = "models/srnet_model_F00_l1_{l1:.0e}_gc_{gc:.0e}.pkl"
# save_file = "models/srnet_model_F00_a1_{a1:.0e}_a2_{a2:.0e}_gc_{gc:.0e}.pkl"


# define hyperparameters
hyperparams = {
    "arch": {
        "in_size": train_data.in_data.shape[1],
        "out_size": train_data.target_data.shape[1],
        "hid_num": (2,0),
        "hid_size": 32, 
        "hid_type": "MLP",
        "lat_size": 16,
        },
    "epochs": 10000,
    "runtime": None,
    "batch_size": 64,
    "lr": 1e-4,
    "wd": 1e-4,
    "l1": 0.0,
    "a1": 0.0,
    "a2": 0.0,
    "gc": 0.0,
    "shuffle": True,
}

def train():
    run_training(SRNet, hyperparams, train_data, val_data, save_file=save_file, device=device, wandb_project=wandb_project)

if __name__ == "__main__":

    # hyperparameter study
    if sweep_id:
        wandb.agent(sweep_id, train, count=sweep_num, project=wandb_project)

    # one training run
    else:
        train()