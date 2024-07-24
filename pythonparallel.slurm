#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=20G
#SBATCH --array=118,114,111,109,106,97,83,71

module purge
module load foss/2022a
module load Python/3.10.4
module load SciPy-bundle/2022.05

Retau=180
k_scale=3
detection_interval=30
read_array_start=1000000
read_array_end=10000000
read_array_interval=10000
whether_16=1

srun -n $SLURM_NTASKS python3 MAIN_pick_bigNonTp_PARALLEL.py 180 $SLURM_ARRAY_TASK_ID 3 7 1000000 10000000 10000 $SLURM_JOB_ID $SLURM_NTASKS 1
