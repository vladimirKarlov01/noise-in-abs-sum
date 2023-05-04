#! /bin/bash
#SBATCH --job-name=eval_run            # Название задачи
#SBATCH --error="eval_run.err"        # Файл для вывода ошибок
#SBATCH --output="eval_run.log"       # Файл для вывода результатов
#SBATCH --time=0-02:30:0            # Максимальное время выполнения
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=4          # Требуемое кол-во CPU

# Executable

### python simpleRun.py -c ./config.json --mixed="../voiceFilter/testData/mixed.wav" --ref="../voiceFilter/testData/ref.wav" --out="./outSpp"
for i in {0..0}
do
python eval_test_model.py \
--dataset-name "aeslc" \
--dataset-path "/home/vakarlov/noise-in-abs-sum/aeslc_MD_filtered_data_0.99/iter_5.hf" \
--model-checkpoint "t5-small" \
--num-workers 4
done
