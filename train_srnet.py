import os
import torch
import joblib
from srnet import SRNet, SRData, run_training
from sdnet import SDNet, SDData
import wandb
from collections.abc import Iterable

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set seeds
seeds = [0, 1, 2, 3, 4]

# set wandb options
wandb_project = "197-gen-train-data-in2"
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
    train_mask = masks['train']
    val_mask = masks['val']
except:
    train_mask = None
    val_mask = None
    print("Warning: No masks for training and validation loaded.")

train_data = SRData(data_path, in_var, lat_var, target_var, data_mask=train_mask, device=device)
val_data = SRData(data_path, in_var, lat_var, target_var, data_mask=val_mask, device=device)

# load discriminator library
fun_path = "funs/F07_v8.lib"
shuffle = True
samples = None
iter_sample = False
    
if fun_path:
    disc_lib = SDData(fun_path, in_var, shuffle=shuffle, samples=samples, iter_sample=iter_sample)
else:
    disc_lib = None

# generate data
lat_size = 5
lat_mean = 3.0
lat_range = 2.0

if target_var is None:
    train_data.generate_functions(disc_lib, lat_size, lat_mean, lat_range)
    val_data.generate_data(train_data.lat_funs, train_data.target_funs)
print(train_data.lat_funs[0])

# define alpha
in_size = train_data.in_data.shape[1]

alpha = []
for j in range(lat_size):
    alpha.append([int(f"{in_var}[:,{i}]" in train_data.lat_funs[0][j]) for i in range(in_size)])
print(alpha)

# set load and save file
load_file = None
disc_file = None
save_file = f"models/srnet_model_G{in_size}{lat_size}_SJNN_MLP_SD_check.pkl"
rec_file = None
log_freq = 25

# define hyperparameters
hyperparams = {
        "arch": {
            "in_size": in_size,
            "lat_size": (lat_size, 1),
            "cell_type": ("SJNN", "MLP"),
            "hid_num": (4, 0),
            "hid_size": 32,
            "cell_kwargs": {
                "alpha": alpha,
                "norm": None,
                "prune": None,
                # "lin_trans": False,
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

def train():
    for seed in seeds if isinstance(seeds, Iterable) else [seeds]:
        run_training(model_cls, hyperparams, train_data, val_data, seed, disc_cls, disc_lib, load_file, disc_file, save_file, rec_file, log_freq, wandb_project, device)

if __name__ == "__main__":

    # hyperparameter study
    if sweep_id:
        wandb.agent(sweep_id, train, count=sweep_num, project=wandb_project)

    # one training run
    else:
        train()