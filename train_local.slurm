#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --time=0-01:00:00

# Load required modules
module load gcc/8.3.0; module load scipy-bundle/2019.10-python-3.7.4
source ~/virtualenv/py37/bin/activate

# Launch python code
python ./train_local.py
