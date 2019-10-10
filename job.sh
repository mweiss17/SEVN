#!/bin/bash
#SBATCH --cpus-per-task=6         # Ask for 6 CPUs
#SBATCH --gres=gpu:1              # Ask for 1 GPU
#SBATCH --mem=10G                 # Ask for 10 GB of RAM
#SBATCH --time=0:10:00            # The job will run for 10 minutes

mkdir $SCRATCH/trained_models
mkdir $SCRATCH/trained_models/ppo/

# 1. Copy your container on the compute node
rsync -avz $SCRATCH/SEVN_latest.sif $SLURM_TMPDIR
# 2. Copy your code on the compute node
rsync -avz $SCRATCH/SEVN-model $SLURM_TMPDIR

seed="$(find $SCRATCH/trained_models/ppo/ -maxdepth 0 -type d | wc -l)"

echo "$(nvidia-smi)"

# 3. Executing your code with singularity
# try singularity run
singularity exec --nv \
        -H $HOME:/home \
        -B $SLURM_TMPDIR:/dataset/ \
        -B $SCRATCH:/final_log/ \
        $SLURM_TMPDIR/SEVN_latest.sif \
        python3 SEVN-model/main.py \
          --env-name "SEVN-Mini-All-Shaped-v1" \
          --custom-gym SEVN_gym \
          --algo ppo \
          --use-gae \
          --lr 5e-4 \
          --clip-param 0.1 \
          --value-loss-coef 0.5 \
          --num-processes 4 \
          --num-steps 128 \
          --num-mini-batch 4 \
          --log-interval 1 \
          --use-linear-lr-decay \
          --entropy-coef 0.01 \
          --comet mweiss17/navi-corl-2019/UcVgpp0wPaprHG4w8MFVMgq7j \
          --seed $seed \
          --num-env-steps 50000000

# 4. Copy whatever you want to save on $SCRATCH
rsync -avz $SLURM_TMPDIR/trained_models $SCRATCH
