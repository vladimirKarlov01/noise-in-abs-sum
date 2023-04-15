#! /bin/bash
#SBATCH --job-name=iter_1            # Название задачи
#SBATCH --error="iter_1.err"        # Файл для вывода ошибок
#SBATCH --output="iter_1.log"       # Файл для вывода результатов
#SBATCH --time=0-0:10:0            # Максимальное время выполнения
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=1          # Требуемое кол-во CPU

# Executable

### python simpleRun.py -c ./config.json --mixed="../voiceFilter/testData/mixed.wav" --ref="../voiceFilter/testData/ref.wav" --out="./outSpp"
for i in {0..0}
do
python conduct_experiment.py \
--dataset-path "/home/vakarlov/noise_in_abs_sum/aeslc_filtered_data_0.99/iter_1.hf" \
--model-checkpoint "t5-small"
done
