#! /bin/bash
#SBATCH --job-name=filtered            # Название задачи
#SBATCH --error="filtered.err"        # Файл для вывода ошибок
#SBATCH --output="filtered.log"       # Файл для вывода результатов
#SBATCH --time=0-01:00:0            # Максимальное время выполнения
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

# Executable

### python simpleRun.py -c ./config.json --mixed="../voiceFilter/testData/mixed.wav" --ref="../voiceFilter/testData/ref.wav" --out="./outSpp"
for i in {0..0}
do
python conduct_experiment.py \
--dataset-name "aeslc" \
--dataset-path "/home/vakarlov/noise_in_abs_sum/aeslc_filtered_data_0.99/iter_1.hf" \
--model-checkpoint "t5-small"
done
