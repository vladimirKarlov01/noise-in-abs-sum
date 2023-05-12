from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
import pandas as pd
import numpy as np
import torch
import random
import os
import time

CACHE_DIR_PATH = "/home/vakarlov/hf-cache-dir"
os.environ["HF_DATASETS_OFFLINE"] = "1"

SEED = 42
def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    set_random_seed(SEED)  
    print('Programm started')
    dataset_name = 'noisy_aeslc'

    if dataset_name == 'xsum_sample':
        start = time.time()
        print('SEED FIXED', time.time() - start)
        raw_data = load_dataset("xsum", cache_dir=CACHE_DIR_PATH)
        print('XSUM LOADED', time.time() - start)
        train_df = pd.DataFrame(raw_data['train']).sample(20000)
        print('PD1 Created', time.time() - start)
        val_df = pd.DataFrame(raw_data['validation']).sample(1500)
        print('PD2 Created', time.time() - start)
        test_df = pd.DataFrame(raw_data['test']).sample(1500)
        print('PD3 Created', time.time() - start)
        raw_data['train'] = Dataset.from_pandas(train_df)
        print('HF1 Created', time.time() - start)
        raw_data['validation'] = Dataset.from_pandas(val_df)
        print('HF2 Created', time.time() - start)
        raw_data['test'] = Dataset.from_pandas(test_df)      
        print('HF3 Created', time.time() - start)
        raw_data = raw_data.rename_column("document", "text")     
        raw_data.save_to_disk(f"{CACHE_DIR_PATH}/xsum_sample.hf")
    elif dataset_name == 'noisy_xsum_sample':
        raw_data = load_from_disk(f"{CACHE_DIR_PATH}/xsum_sample.hf")
        train_df = pd.DataFrame(raw_data['train']).drop(columns=['__index_level_0__'])

        print('raw_data loaded')

        ood_data = load_dataset('aeslc', cache_dir=CACHE_DIR_PATH).rename_column("email_body", "text").rename_column("subject_line", "summary")
        ood_data = pd.DataFrame(ood_data['train'])

        print('ood_data loaded')

        train_df = pd.concat([train_df, ood_data.sample(raw_data['train'].num_rows // 10)])
        raw_data['train'] = Dataset.from_pandas(train_df)
        raw_data.save_to_disk(f"{CACHE_DIR_PATH}/{dataset_name}.hf")
    elif dataset_name == 'noisy_aeslc':
        raw_data = load_dataset('aeslc', cache_dir=CACHE_DIR_PATH).rename_column("email_body", "text").rename_column("subject_line", "summary")
        train_df = pd.DataFrame(raw_data['train'])

        print('raw_data loaded')

        ood_data = load_from_disk(f"{CACHE_DIR_PATH}/xsum_sample.hf")
        ood_data = pd.DataFrame(ood_data['train']).drop(columns=['__index_level_0__'])

        print('ood_data loaded')

        train_df = pd.concat([train_df, ood_data.sample(raw_data['train'].num_rows // 10)])
        raw_data['train'] = Dataset.from_pandas(train_df)
        raw_data.save_to_disk(f"{CACHE_DIR_PATH}/{dataset_name}.hf")