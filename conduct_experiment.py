# хотим функцию, принимающую на вход путь к датасету + модель, которую учим (чекпоинт)

import os
import warnings
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
import evaluate
import wandb
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from functools import partial

def preprocess_function(examples, tokenizer, checkpoint_name):
    if checkpoint_name.split('/')[-1].split('-')[0] == 't5':
        prefix = "summarize: "
    else:
        prefix = ''

    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length')

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer, rouge):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def conduct_experiment(hf_df_path, checkpoint_name):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenized_orig_data = raw_data.map(partial(preprocess_function, tokenizer=tokenizer,
                                               checkpoint_name=checkpoint_name), batched=True)
    rouge = evaluate.load("rouge", cache_dir="/home/vakarlov/hf-cache-dir/rouge")
    original_run_name = dataset_name + "_original_" + checkpoint_name
    wandb.login()
    wandb.init(project='noise-in-abs-sum', name=original_run_name)
    original_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=original_model)
    batch_size, eval_batch_size = 16, 32  # V100 config
    # batch_size, eval_batch_size = 8, 16  # colab / local config

    num_workers = 8

    training_args = Seq2SeqTrainingArguments(
        output_dir=dataset_name + "-original-" + checkpoint_name,
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
        model=original_model,
        args=training_args,
        train_dataset=tokenized_orig_data["train"],
        eval_dataset=tokenized_orig_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ! TEST PREDICTION POWERED PY AKIM (BATYA) !

    wandb.finish()


if __name__ == '__main__':
    conduct_experiment(hf_df_path='', checkpoint_name='t5-small')
