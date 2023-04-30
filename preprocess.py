
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