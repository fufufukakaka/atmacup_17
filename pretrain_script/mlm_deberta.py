import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class CFG:
    seed = 42
    model_name = "microsoft/deberta-v3-large"
    epochs = 3
    batch_size = 2
    lr = 1e-6
    weight_decay = 1e-6
    max_len = 512
    mask_prob = 0.15
    n_accumulate = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=CFG.seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def preprocessing(df):
    df["Title"] = df["Title"].fillna("")
    df["Review Text"] = df["Review Text"].fillna("")
    df["prompt"] = df["Title"] + " " + df["Review Text"]
    return df


class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized_data = self.tokenizer.encode_plus(
            text,
            max_length=CFG.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized_data["input_ids"].squeeze(),
            "attention_mask": tokenized_data["attention_mask"].squeeze(),
        }


def main():
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForMaskedLM.from_pretrained(CFG.model_name)

    train_df = pd.read_csv("data/train.csv")
    train_df = preprocessing(train_df)
    test_df = pd.read_csv("data/test.csv")
    test_df = preprocessing(test_df)
    train_texts = train_df["prompt"].tolist()
    test_texts = test_df["prompt"].tolist()
    all_texts = train_texts + test_texts

    dataset = MLMDataset(all_texts, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=CFG.mask_prob
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=CFG.epochs,
        per_device_train_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.n_accumulate,
        learning_rate=CFG.lr,
        weight_decay=CFG.weight_decay,
        save_total_limit=2,
        save_steps=1000,
        logging_dir="./logs",
        logging_steps=500,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("./pretrained_models")


if __name__ == "__main__":
    main()
