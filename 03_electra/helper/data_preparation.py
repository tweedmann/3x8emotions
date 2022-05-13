import re
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    random_split,
)


class HateDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def getTempFiles(
    path_train,
    path_test,
    path_tmp,
    path_validation=None,
    frac_train=1.0,
    frac_val=0.1,
    seed=123,
):
    Path(path_tmp).mkdir(parents=True, exist_ok=True)
    path_tmp_train = Path(path_tmp) / "train.csv"
    path_tmp_validation = Path(path_tmp) / "validation.csv"
    path_tmp_test = Path(path_tmp) / "test.csv"

    df = pd.read_csv(path_train, encoding="utf-8", sep="\t")
    df = df.sample(frac=frac_train, random_state=seed)
    if path_validation is None:
        df_train, df_validation = train_test_split(
            df, test_size=frac_val, random_state=seed
        )
    else:
        df_train = df
        df_validation = pd.read_csv(path_validation, encoding="utf-8", sep="\t")
        df_validation = df_validation.sample(
            n=int(frac_val * len(df_train)), random_state=seed
        )

    df_test = pd.read_csv(path_test, encoding="utf-8", sep="\t")

    df_train.to_csv(path_tmp_train, encoding="utf-8", sep="\t", index=False)
    df_validation.to_csv(path_tmp_validation, encoding="utf-8", sep="\t", index=False)
    df_test.to_csv(path_tmp_test, encoding="utf-8", sep="\t", index=False)

    print(len(df_train), len(df_validation), len(df_test))
    return path_tmp_train, path_tmp_validation, path_tmp_test


def getHateDatasets(data_params, selected_dataset, tokenizer):
    label_name = data_params[selected_dataset]["label"]
    text_name = data_params[selected_dataset]["text"]
    
    def preprocess(row):
        # preprocess text
        row["text"] = re.sub(r"\|lbr\||\|LBR\||\|AMP\||&gt;|&amp;", " ", row["text"])
        row["text"] = re.sub(r"(^|\s)@[A-Za-z0-9_-]*", " ", row["text"])
        # convert string label to integer
        # mapping is stored in model_params
        selected = data_params[selected_dataset]
        row[selected["label"]] = selected["mapping"][row[selected["label"]]]
        return row
    
    #print(data_params[selected_dataset]["train"])
    #print(data_params[selected_dataset]["validation"])
    #print(data_params[selected_dataset]["test"])
    # load data
    dataset = load_dataset(
        "csv",
        data_files={
            "train": data_params[selected_dataset]["train"],
            "validation": data_params[selected_dataset]["validation"],
            "test": data_params[selected_dataset]["test"],
        },
        delimiter="\t",
    )

    # preprocess data
    dataset["train"] = dataset["train"].map(preprocess)
    dataset["validation"] = dataset["validation"].map(preprocess)
    dataset["test"] = dataset["test"].map(preprocess)

    # tokenize data
    train_encodings = tokenizer(
        dataset["train"][text_name], truncation=True, padding=True
    )
    val_encodings = tokenizer(
        dataset["validation"][text_name], truncation=True, padding=True
    )
    test_encodings = tokenizer(
        dataset["test"][text_name], truncation=True, padding=True
    )

    train_dataset = HateDataset(train_encodings, dataset["train"][label_name])
    val_dataset = HateDataset(val_encodings, dataset["validation"][label_name])
    test_dataset = HateDataset(test_encodings, dataset["test"][label_name])

    return train_dataset, val_dataset, test_dataset


def getHateDataLoaders(data_params, selected_dataset, tokenizer, batch_size=128):
    train_dataset, val_dataset, test_dataset = getHateDatasets(
        data_params, selected_dataset, tokenizer
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(train_dataset),
        num_workers=8,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(val_dataset),
        num_workers=8,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=8,
    )

    return train_loader, val_loader, test_loader

