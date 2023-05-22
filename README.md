# noise-in-abs-sum
Репозиторий посвящен ВКР Карлова В.А. на тему "Noise Filtering in Abstractive Summarization".

## Аннотация
Новейшие модели, решающие задачу абстрактивной суммаризации текстов, обучаются на очень больших корпусах, собранных, в основном, путем ивзлечения информации с интернет страниц. По этой причине большинство суммаризационных датасетов (таких как AESLC, XSum и другие) могут содержать значительное количество ошибок различного рода, а также шумных примеров. В данной работе мы предлагаем два алгоритма фильтрации выбросов, которые улучшают качество суммаризации (в терминах автоматических метрик ROUGE, BLEU и другиx) для моделей, обученных заново на данных без шума. Для изучения границ применимости предложенных подходов, мы также приводим два способа искусственной генерации шума и тестируем методы на зашумленных данных.

## Annotation
State-of-the-art models for summarization require training on big amounts of data, which is mostly collected from the web. This is the main reason why a lot of summarization datasets (e.g. AESLC, XSum and others) may contain noisy and mislabeled examples. In this study, we propose two algorithms for noise filtration, increasing the values of automatic metrics like ROUGE for models, retrained on filtered data. To investigate the limits of applicability of the proposed approaches, we also present two methods for creating synthetic noise and test the methods on noisy data.

## Структура репозитория
* Алгоритмы фильтрации и создания шума доступны в jupyter-ноутбуке -- **"noise-in-abs-sum.ipynb"**.
* Пайплайн для обучения моделей, принимающий на вход датесет (отфильтрованный или оригинальный) и конфигурацию модели -- **"train.py"**.
* Пайплайн для тестирования обученной модели -- **"eval_test_model.py"**.
* Фреймворк для вычисления метрик качества суммаризации -- **"metrics/compute_metrics.py"**.
* Требования к установленным библлиоткекам -- **"requirements.txt"**.
