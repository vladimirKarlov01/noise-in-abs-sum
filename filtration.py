def load_hf_dataset(dataset_name):
    raw_data = load_dataset(dataset_name, cache_dir="/home/vakarlov/hf-cache-dir/")

    if dataset_name == "aeslc":
        raw_data = raw_data.rename_column("email_body", "text").rename_column("subject_line", "summary")
    elif dataset_name == "xsum":
        raw_data = raw_data.rename_column("document", "text")
    else:
        raise NotImplementedError('Unknown dataset')
    return raw_data