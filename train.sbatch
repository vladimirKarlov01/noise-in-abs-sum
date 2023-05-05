#! /bin/bash
#SBATCH --job-name=train_run            # Название задачи
#SBATCH --error="train_run.err"        # Файл для вывода ошибок
#SBATCH --output="train_run.log"       # Файл для вывода результатов
#SBATCH --time=0-01:30:0            # Максимальное время выполнения
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
# SBATCH --cpus-per-task=6         # Требуемое кол-во CPU

# Executable

### python simpleRun.py -c ./config.json --mixed="../voiceFilter/testData/mixed.wav" --ref="../voiceFilter/testData/ref.wav" --out="./outSpp"
for i in {0..0}
do
python train.py \
--dataset-name "xsum_sample" \
--dataset-path "/home/vakarlov/noise-in-abs-sum/filtered_data/xsum_sample_LOSS_0.99.hf" \
--run-name "xsum_sample_LOSS_0.99" \
--model-checkpoint "t5-small" \
--num-workers 6
done
