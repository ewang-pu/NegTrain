#!/bin/bash 
#SBATCH --job-name=train         # TODO create a short name for your job
#SBATCH --nodes=1              # node count (default is 1 should work)
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00          # TODO set total run time limit (HH:MM:SS) (default 1-3 hrs could work)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=evanwang@princeton.edu # TODO add netid if you want emails, or delete these past 3 lines


module purge
module load anaconda3/2023.9
module load cudatoolkit/12.2
conda activate NegTrain # TODO your conda environment 

CUDA_VISIBLE_DEVICES=0
python -u src/training/main.py \
    --train-data="./temp_data/train_neg_clip.tsv" \
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