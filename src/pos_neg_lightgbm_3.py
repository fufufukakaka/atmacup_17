import os
import warnings

from sklearn.model_selection import StratifiedKFold

import wandb

warnings.filterwarnings("ignore")
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import pipeline

sentiment_pipe = pipeline(
    "text-classification", model="finiteautomata/bertweet-base-sentiment-analysis"
)

os.environ["WANDB_PROJECT"] = "atmacup_17"
os.environ["WANDB_LOG_MODEL"] = "false"


class CFG:
    VER = 3
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
    # clothing_master_df と merge
    df = df.merge(clothing_master_df, on="Clothing ID", how="left")
    # カテゴリを数値に変換
    le = LabelEncoder()
    df["Division Name"] = le.fit_transform(df["Division Name"])
    le = LabelEncoder()
    df["Department Name"] = le.fit_transform(df["Department Name"])
    le = LabelEncoder()
    df["Class Name"] = le.fit_transform(df["Class Name"])

    df["Title"] = df["Title"].fillna("")
    df["Review Text"] = df["Review Text"].fillna("")
    df["prompt"] = df["Title"] + " " + df["Review Text"]

    # prompt から sentiment (positive score)を抽出
    scores = []
    for _title in tqdm(df["Title"].values):
        _res = sentiment_pipe(_title)
        scores.append(_res[0]["score"])

    df["title_positive_score"] = scores

    return df


def main():
    wandb.init(project="atmacup_17", name=f"lightgbm_{CFG.VER}")
    seed_everything(CFG.SEED)

    clothing_master_df = pd.read_csv(CFG.DATA_PATH / "clothing_master.csv")
    train_df = pd.read_csv(CFG.DATA_PATH / "train.csv")
    test_df = pd.read_csv(CFG.DATA_PATH / "test.csv")

    train_df = preprocessing(train_df, clothing_master_df)
    test_df = preprocessing(test_df, clothing_master_df)
    train_df["labels"] = train_df[CFG.target_col].astype(np.int8)

    predictions = np.zeros((len(train_df), 1))

    kfold = StratifiedKFold(n_splits=CFG.N_SPLIT, shuffle=True, random_state=CFG.SEED)
    for fold, (train_index, valid_index) in enumerate(
        kfold.split(train_df, train_df["Rating"])
    ):
        fold_train = train_df.iloc[train_index][
            [
                "Age",
                "Positive Feedback Count",
                "Division Name",
                "Department Name",
                "Class Name",
                "Recommended IND",
                "title_positive_score",
            ]
        ].copy()
        fold_eval = train_df.iloc[valid_index][
            [
                "Age",
                "Positive Feedback Count",
                "Division Name",
                "Department Name",
                "Class Name",
                "Recommended IND",
                "title_positive_score",
            ]
        ].copy()

        lgb_train = lgb.Dataset(
            fold_train.drop(columns=["Recommended IND"]),
            label=fold_train["Recommended IND"],
        )
        lgb_eval = lgb.Dataset(
            fold_eval.drop(columns=["Recommended IND"]),
            label=fold_eval["Recommended IND"],
            reference=lgb_train,
        )

        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_eval,
            num_boost_round=100,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=10, verbose=True
                ),  # early_stopping用コールバック関数
                lgb.log_evaluation(0),
            ],  # コマンドライン出力用コールバック関数
        )

        _predict = model.predict(
            fold_eval.drop(columns=["Recommended IND"]),
            num_iteration=model.best_iteration,
        )
        predictions[valid_index] = _predict.reshape(-1, 1)

        model.save_model(
            f"{CFG.MODEL_DATA_PATH}/lightgbm-seed{CFG.SEED}-Ver{CFG.VER}_fold{fold}.txt"
        )

    # 予測結果をデータフレームに追加(学習データの AUC を計算)
    train_df["preds"] = predictions
    auc = roc_auc_score(train_df["Recommended IND"], train_df["preds"])
    wandb.log({"AUC": auc})

    # テストデータで予測
    test_predictions = np.zeros((len(test_df), 1))

    for fold in range(CFG.N_SPLIT):
        model = lgb.Booster(
            model_file=f"{CFG.MODEL_DATA_PATH}/lightgbm-seed{CFG.SEED}-Ver{CFG.VER}_fold{fold}.txt"
        )
        test_predictions += model.predict(
            test_df[
                [
                    "Age",
                    "Positive Feedback Count",
                    "Division Name",
                    "Department Name",
                    "Class Name",
                    "title_positive_score",
                ]
            ]
        ).reshape(-1, 1)

    test_df["preds"] = test_predictions / CFG.N_SPLIT

    # 予測結果を保存(submission)
    submission = pd.DataFrame()
    submission["target"] = test_df["preds"]

    submission.to_csv(
        f"./predictions/lightgbm-seed{CFG.SEED}-Ver{CFG.VER}_test_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
