#!/bin/bash

#SBATCH --job-name=job3          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                # memory per node (4G per cpu-core is default)
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=fanyi@princeton.edu

module purge
module load anaconda3
conda activate cos429

python3 part2_lenet6.py