
# Activate Conda
VAR_USER=kumarsh
eval "$(/cluster/home/${VAR_USER}/miniconda3/bin/conda shell.bash hook)"
conda activate cil_env

cd /cluster/scratch/$VAR_USER/cil_project

python trainingutil.py --config experiment_configs/exp_01_resnet50_full.yaml