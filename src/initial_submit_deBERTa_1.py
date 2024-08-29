import os
import warnings

import wandb

warnings.filterwarnings("ignore")
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

os.environ["WANDB_PROJECT"] = "atmacup_17"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


class CFG:
    VER = 1
    AUTHOR = "fufufukakaka"
    COMPETITION = "atmacup17"
    DATA_PATH = Path("data")
    OOF_DATA_PATH = Path("oof")
    MODEL_DATA_PATH = Path("models")
    MODEL_PATH = "microsoft/deberta-v3-large"
    MAX_LENGTH = 256
    STEPS = 25
    USE_GPU = torch.cuda.is_available()
    SEED = 0
    N_SPLIT = 2
    target_col = "Recommended IND"
    target_col_class_num = 2
    metric = "auc"
    metric_maximize_flag = True


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if CFG.USE_GPU:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocessing(df, clothing_master_df):
    df["Title"] = df["Title"].fillna("")
    df["Review Text"] = df["Review Text"].fillna("")
    df["prompt"] = df["Title"] + " " + df["Review Text"]
    return df


tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_PATH)


def tokenize(sample):
    return tokenizer(sample["prompt"], max_length=CFG.MAX_LENGTH, truncation=True)


# metricをAUCに変更
def compute_metrics(p):
    preds, labels = p
    preds = torch.softmax(torch.tensor(preds), dim=1).numpy()
    score = roc_auc_score(labels, preds[:, 1])
    return {"auc": score}


def predict_on_test(test_df, model_path, tokenizer, clothing_master_df):
    # モデルとトークナイザーをロード
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # テストデータの前処理
    test_df = preprocessing(test_df, clothing_master_df)

    # トークン化
    test_df["prompt"] = test_df["Title"] + " " + test_df["Review Text"]
    test_dataset = Dataset.from_pandas(test_df[["prompt"]].copy())
    test_dataset = test_dataset.map(tokenize).remove_columns(
        ["prompt", "__index_level_0__"]
    )

    # Trainerを使って予測
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./temp", per_device_eval_batch_size=4, no_cuda=not CFG.USE_GPU
        ),
        tokenizer=tokenizer,
    )

    # 予測の実行
    predictions = trainer.predict(test_dataset).predictions
    predictions = torch.softmax(torch.tensor(predictions), dim=1).numpy()

    # 予測結果をデータフレームに追加
    submission = pd.DataFrame()
    submission["target"] = predictions[:, 1]

    # 予測結果をCSVファイルに保存
    submission.to_csv(
        f"./predictions/deberta-large-seed{CFG.SEED}-Ver{CFG.VER}_test_predictions.csv",
        index=False,
    )


def main():
    seed_everything(CFG.SEED)

    clothing_master_df = pd.read_csv(CFG.DATA_PATH / "clothing_master.csv")
    train_df = pd.read_csv(CFG.DATA_PATH / "train.csv")
    test_df = pd.read_csv(CFG.DATA_PATH / "test.csv")

    train_df = preprocessing(train_df, clothing_master_df)
    test_df = preprocessing(test_df, clothing_master_df)
    train_df["labels"] = train_df[CFG.target_col].astype(np.int8)

    predictions = np.zeros((len(train_df), CFG.target_col_class_num))

    kfold = StratifiedKFold(n_splits=CFG.N_SPLIT, shuffle=True, random_state=CFG.SEED)
    for fold, (train_index, valid_index) in enumerate(
        kfold.split(train_df, train_df["Rating"])
    ):
        ds_train = Dataset.from_pandas(
            train_df.iloc[train_index][["prompt", "labels"]].copy()
        )
        ds_eval = Dataset.from_pandas(
            train_df.iloc[valid_index][["prompt", "labels"]].copy()
        )

        ds_train = ds_train.map(tokenize).remove_columns(
            ["prompt", "__index_level_0__"]
        )
        ds_eval = ds_eval.map(tokenize).remove_columns(["prompt", "__index_level_0__"])

        train_args = TrainingArguments(
            output_dir=f"{CFG.MODEL_DATA_PATH}/deberta-large-fold{fold}",
            fp16=True,
            learning_rate=2e-5,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            report_to="none",
            evaluation_strategy="steps",
            do_eval=True,
            eval_steps=CFG.STEPS,
            save_total_limit=3,
            save_strategy="steps",
            save_steps=CFG.STEPS,
            logging_steps=CFG.STEPS,
            lr_scheduler_type="linear",
            metric_for_best_model="auc",  # AUCを評価に使用する
            greater_is_better=True,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_safetensors=True,
            seed=CFG.SEED,
            data_seed=CFG.SEED,
        )

        config = AutoConfig.from_pretrained(CFG.MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(
            CFG.MODEL_PATH, config=config
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=ds_train,
            eval_dataset=ds_eval,
            data_collator=DataCollatorWithPadding(tokenizer),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.save_model(
            f"deberta-large-seed{CFG.SEED}-Ver{CFG.VER}/deberta-large-fold{fold}"
        )
        tokenizer.save_pretrained(
            f"deberta-large-seed{CFG.SEED}-Ver{CFG.VER}/deberta-large-fold{fold}"
        )

        predictions[valid_index] = torch.softmax(
            torch.tensor(trainer.predict(ds_eval).predictions), dim=1
        ).numpy()

    train_df[f"deberta_large_Ver{CFG.VER}_pred_prob"] = predictions[:, 1]
    train_df.to_csv(f"./oof/deberta-large-seed{CFG.SEED}-Ver{CFG.VER}.csv", index=False)

    _auc = roc_auc_score(
        train_df["labels"], train_df[f"deberta_large_Ver{CFG.VER}_pred_prob"]
    )
    wandb.log({"OOF AUC": _auc})

    predict_on_test(test_df, CFG.MODEL_PATH, tokenizer, clothing_master_df)


if __name__ == "__main__":
    main()
