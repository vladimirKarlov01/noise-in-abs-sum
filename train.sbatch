#! /bin/bash
#SBATCH --job-name=train_run            # Название задачи
#SBATCH --error="train_run.err"        # Файл для вывода ошибок
#SBATCH --output="train_run.log"       # Файл для вывода результатов
#SBATCH --time=0-01:30:0            # Максимальное время выполнения
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
# SBATCH --cpus-per-task=6         # Требуемое кол-во CPU

# Executable

for i in {0..0}
do
python train.py \
--dataset-name "shuffled_aeslc_0.1" \
--dataset-path "/home/vakarlov/noise-in-abs-sum/filtered_data/shuffled_aeslc_0.1_LOSS_0.9.hf" \
--run-name "shuffled_aeslc_0.1_LOSS_0.9" \
--model-checkpoint "t5-small" \
--num-workers 6 \
--seed 42

python train.py \
--dataset-name "shuffled_aeslc_0.1" \
--dataset-path "/home/vakarlov/noise-in-abs-sum/filtered_data/shuffled_aeslc_0.1_LOSS_0.9.hf" \
--run-name "shuffled_aeslc_0.1_LOSS_0.9" \
--model-checkpoint "t5-small" \
--num-workers 6 \
--seed 43

python train.py \
--dataset-name "shuffled_aeslc_0.1" \
--dataset-path "/home/vakarlov/noise-in-abs-sum/filtered_data/shuffled_aeslc_0.1_LOSS_0.9.hf" \
--run-name "shuffled_aeslc_0.1_LOSS_0.9" \
--model-checkpoint "t5-small" \
--num-workers 6 \
--seed 44

python train.py \
--dataset-name "shuffled_aeslc_0.1" \
--dataset-path "/home/vakarlov/noise-in-abs-sum/filtered_data/shuffled_aeslc_0.1_LOSS_0.9.hf" \
--run-name "shuffled_aeslc_0.1_LOSS_0.9" \
--model-checkpoint "t5-small" \
--num-workers 6 \
--seed 45

python train.py \
--dataset-name "shuffled_aeslc_0.1" \
--dataset-path "/home/vakarlov/noise-in-abs-sum/filtered_data/shuffled_aeslc_0.1_LOSS_0.9.hf" \
--run-name "shuffled_aeslc_0.1_LOSS_0.9" \
--model-checkpoint "t5-small" \
--num-workers 6 \
--seed 46
done
