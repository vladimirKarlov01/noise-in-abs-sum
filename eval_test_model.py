import os
import warnings
import json

import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_from_disk

from functools import partial
from argparse import ArgumentParser
from pathlib import Path

from metrics.compute_metrics import compute_metrics as compute_metrics_test
from preprocess import preprocess_function

CACHE_DIR_PATH = "/home/vakarlov/hf-cache-dir"

def eval_test(dataset_name, hf_df_path, checkpoint_name, num_workers=6):
    filtered_data = load_from_disk(hf_df_path)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    tokenized_data = filtered_data['test'].map(partial(preprocess_function, tokenizer=tokenizer,
                                               checkpoint_name=checkpoint_name), batched=True)
    
    run_name = f"{hf_df_path.parts[-2]}_{hf_df_path.stem}"

    model = AutoModelForSeq2SeqLM.from_pretrained(f"results/{run_name}/final_checkpoint/")
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    batch_size, eval_batch_size = 32, 64  # V100 config

    # ================================================================================
    # TEST PREDICTION
    test_args = Seq2SeqTrainingArguments(
        output_dir=f"results/{run_name}",
        overwrite_output_dir = False,
        dataloader_num_workers=num_workers,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        predict_with_generate=True,
        include_inputs_for_metrics = True,
        report_to="wandb",
        run_name=run_name + '_test',
    )
    trainer_test = Seq2SeqTrainer(
        model=model,
        args=test_args,
        tokenizer=tokenizer,
        eval_dataset=tokenized_data,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics_test, tokenizer=tokenizer)
    )

    test_results = trainer_test.evaluate()
    print(test_results)

    with open(f"results/{run_name}/test_metrics.json", "w") as out_file:
        json.dump(test_results, out_file)
    # ================================================================================


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_PROJECT"] = "noise-in-abs-sum"
    
    parser = ArgumentParser()
    # argument groups are useful for separating semantically different parameters
    data_group = parser.add_argument_group("args")
    data_group.add_argument(
        "--dataset-name", type=str, help="HF dataset name"
    )
    data_group.add_argument(
        "--dataset-path", type=Path, help="Path to the filtered dataset in HF format"
    )
    data_group.add_argument(
        "--model-checkpoint", type=str, help="HF model checkpoint"
    )
    data_group.add_argument(
        "--num-workers", type=int, help="PyTorch dataloader num workers"
    )
    args = parser.parse_args()

    eval_test(dataset_name=args.dataset_name,
              hf_df_path=args.dataset_path,
              checkpoint_name=args.model_checkpoint,
              num_workers=args.num_workers)