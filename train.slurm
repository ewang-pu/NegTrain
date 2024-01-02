#!/bin/bash 
#SBATCH --job-name=train         # TODO create a short name for your job
#SBATCH --nodes=1              # node count (default is 1 should work)
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00          # TODO set total run time limit (HH:MM:SS) (default 1-3 hrs could work)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=evanwang@princeton.edu # TODO add netid if you want emails, or delete these past 3 lines


module purge
module load anaconda3/2023.9
conda activate NegTrain # TODO your conda environment 

CUDA_VISIBLE_DEVICES=0 python -m training.main \
    --train-data="./temp_data/train_neg_clip.tsv" \
    --report-to tensorboard \
    --batch-size=256 \
    --epochs=5 \
    --name="negclip_256_1e-6" \
    --lr=1e-6 \
    --val-data="./temp_data/valid_neg_clip.tsv"  \
    --logs="./logs/negCLIP/" \
    --pretrained="openai" \
    --model="ViT-B-32" \
    --workers 14 \
    --warmup 50