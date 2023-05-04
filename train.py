import os
import warnings

import numpy as np
import torch

from transformers import AutoTokenizer
import evaluate
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_from_disk
import wandb

from functools import partial
from argparse import ArgumentParser
from pathlib import Path

from preprocess import preprocess_function

CACHE_DIR_PATH = "/home/vakarlov/hf-cache-dir"
 

def compute_metrics(eval_pred, tokenizer, rouge):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def conduct_experiment(dataset_name, hf_df_path, checkpoint_name, num_workers=6):
    # хотим функцию, принимающую на вход путь к датасету + модель, которую учим (чекпоинт)
    filtered_data = load_from_disk(hf_df_path)
    # filtered_data = load_dataset('aeslc')['train'].rename_column("email_body", "text").rename_column("subject_line", "summary")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    tokenized_data = filtered_data.map(partial(preprocess_function, tokenizer=tokenizer,
                                               checkpoint_name=checkpoint_name), batched=True)

    rouge = evaluate.load("rouge", cache_dir=CACHE_DIR_PATH)

    run_name = f"{hf_df_path.parts[-2]}_{hf_df_path.stem}"

    wandb.login(key='da5db6589b225ae96b7ef486f82d4061f936a80b')
    wandb.init(project='noise-in-abs-sum', name=run_name)
    wandb.log({'hyperparams/retained_obj_num': filtered_data['train'].num_rows,
               # 'hyperparams/threshold' : THRESHOLD,
               # 'hyperparams/quantile': quantile,
               'hyperparams/checkpoint_name': checkpoint_name,
               'hyperparams/dataset_name': dataset_name})

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_name)
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    batch_size, eval_batch_size = 32, 64  # V100 config

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"results/{run_name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        report_to="wandb",
        dataloader_num_workers=num_workers,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer, rouge=rouge),
    )

    trainer.train()
    trainer.model.save_pretrained(f"results/{run_name}/final_checkpoint/")

    wandb.finish()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    conduct_experiment(dataset_name=args.dataset_name,
                       hf_df_path=args.dataset_path,
                       checkpoint_name=args.model_checkpoint,
                       num_workers=args.num_workers)
