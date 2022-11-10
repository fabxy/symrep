#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --time=0-02:00:00

# Load required modules
if [ $PAWSEY_CLUSTER == "setonix" ]; then
    module load python/3.10.4
    source ~/venv/py310/bin/activate
else
    module load gcc/8.3.0; module load scipy-bundle/2019.10-python-3.7.4
    source ~/virtualenv/py37/bin/activate
fi

# Launch python code
python ./train_srnet_MLP_F07_v2.py
