#!/bin/bash 
#SBATCH --job-name=gather        # TODO create a short name for your job
#SBATCH --nodes=1                # node count (default is 1 should work)
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:30:00          # TODO set total run time limit (HH:MM:SS) (default 1-3 hrs could work)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=evanwang@princeton.edu # TODO add netid if you want emails, or delete these past 3 lines

module purge
module load anaconda3/2023.9
conda activate NegTrain # TODO your conda environment 

python3 src/data/gather_cc.py local_data/Train_GCC-training.tsv local_data/Validation_GCC-1.1.0-Validation.tsv