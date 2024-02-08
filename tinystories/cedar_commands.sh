#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-1
#SBATCH --output=outputs/Feb08/outputs/output_%j.txt
#SBATCH --gres=gpu:a100:1

source ~/virtual_env_meta_step/bin/activate
module load python/3.10
module load cuda

`sed -n "${SLURM_ARRAY_TASK_ID}p" <outputs/Feb05_05_09_13/export.dat`
echo ${SLURM_ARRAY_TASK_ID}

echo "Current working directory is `pwd`"
echo "Running on hostname `hostname`"

echo "Starting run at: `date`"
python3 train.py --optimizer HF --alg-base Lion --weight-decay-base 1 --normalizer-param-base -1 --momentum-param-base .99 --Lion-beta2-base .9 --alg-meta Lion --meta-stepsize 1e-3 --alpha0 1e-5 --stepsize-groups scalar --weight-decay-meta 0 --normalizer-param-meta -1 --momentum-param-meta .99 --Lion-beta2-meta .9 --seed 0 --gamma 1 --run-name 1 --save-directory outputs/Feb08 --max-time 00:10:00
echo "Program test finished with exit code $? at: `date`"

