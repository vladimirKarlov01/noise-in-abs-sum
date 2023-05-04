from datasets import load_dataset, DatasetDict, Dataset
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
    start = time.time()
    set_random_seed(SEED)    
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
