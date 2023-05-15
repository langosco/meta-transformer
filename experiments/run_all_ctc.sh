#!/bin/bash
set -o errexit

for task in augmentation optimizer activation initialization batch_size 
do
    python train_ctc.py --task $task --use_wandb --epochs 5
done
