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
wandb_project = "196-gen-train-data-in2-lat3"
sweep_id = None
sweep_num = None

# select generator and discriminator
model_cls = SRNet
disc_cls = SDNet

# load discriminator library
fun_path = "funs/F07_v8.lib"
in_var = "X07"
shuffle = True
samples = None
iter_sample = False
    
if fun_path:
    disc_lib = SDData(fun_path, in_var, shuffle=shuffle, samples=samples, iter_sample=iter_sample)
else:
    disc_lib = None

# generate data
data_path = "data_1k"
lat_size = 3
lat_mean = 3.0
lat_range = 2.0

mask_ext = ".mask"
try:
    masks = joblib.load(os.path.join(data_path, in_var + mask_ext))
    train_mask = masks['train']
    val_mask = masks['val']
except:
    train_mask = None
    val_mask = None
    print("Warning: No masks for training and validation loaded.")

train_data = SRData(data_path, in_var, data_mask=train_mask, device=device)
val_data = SRData(data_path, in_var, data_mask=val_mask, device=device)

idxs = torch.randperm(disc_lib.len)[:lat_size]

raw_funs = [[disc_lib.funs[i][0].replace('N0*','')] for i in idxs]
raw_means = disc_lib.evaluate(raw_funs, train_data.in_data).abs().mean(dim=1).squeeze().tolist()

funs = [[disc_lib.funs[idx][0].replace('N0*', f'{1.0/raw_means[i]:.2f}*({lat_range}*U0+{lat_mean-1.0})*')] for i, idx in enumerate(idxs)]

train_data.lat_data = [disc_lib.evaluate(funs, train_data.in_data).squeeze().T]
train_data.target_data = train_data.lat_data[0].sum(dim=1).unsqueeze(-1)

means = train_data.lat_data[0].abs().mean(dim=0).tolist()
eff_funs = [[disc_lib.funs[idx][0].replace('N0', f'{means[i]/raw_means[i]:.2f}')] for i, idx in enumerate(idxs)]
train_data.target_var = ' + '.join([f[0] for f in eff_funs])
print(train_data.target_var)

val_data.lat_data = [disc_lib.evaluate(eff_funs, val_data.in_data).squeeze().T]
val_data.target_data = val_data.lat_data[0].sum(dim=1).unsqueeze(-1)

# get alpha
in_vars = [f"{in_var}[:,{i}]" for i in range(train_data.in_data.shape[1])]

alpha = []
for i in range(lat_size):
    alpha.append([int(var in raw_funs[i][0]) for var in in_vars])
print(alpha)

# set load and save file
load_file = None
disc_file = None
save_file = "models/srnet_model_G23_SJNN_MLP_SD_check.pkl"
rec_file = None
log_freq = 25

# define hyperparameters
hyperparams = {
        "arch": {
            "in_size": train_data.in_data.shape[1],
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