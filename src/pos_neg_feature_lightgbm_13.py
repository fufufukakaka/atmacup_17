import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline
import optuna.integration.lightgbm as opt_lgb

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1,
)

import wandb


def cloth_rating_mean_var(train_df, test_df, cloth_df):
    cloth_df["Rating Mean"] = cloth_df["Clothing ID"].map(
        train_df.groupby("Clothing ID")["Rating"].mean()
    )
    cloth_df["Rating Var"] = cloth_df["Clothing ID"].map(
        train_df.groupby("Clothing ID")["Rating"].var()
    )

    train_df = train_df.merge(cloth_df, how="left", on="Clothing ID")
    test_df = test_df.merge(cloth_df, how="left", on="Clothing ID")

    return train_df, test_df


def main():
    wandb.init(project="atmacup_17", name="lightgbm_13")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    cloth_df = pd.read_csv("data/clothing_master.csv")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_df["text"] = (
        train_df["Title"].fillna("none")
        + "[SEP]"
        + train_df["Review Text"].fillna("none")
    )
    test_df["text"] = (
        test_df["Title"].fillna("none")
        + "[SEP]"
        + test_df["Review Text"].fillna("none")
    )

    pos_scores = []
    neg_scores = []
    neutral_scores = []
    for _title in tqdm(train_df["text"].values):
        _res = distilled_student_sentiment_classifier(_title, max_length=512)
        pos_scores.append(_res[0][0]["score"])
        neutral_scores.append(_res[0][1]["score"])
        neg_scores.append(_res[0][2]["score"])
    train_df["positive_score"] = pos_scores
    train_df["neutral_score"] = neutral_scores
    train_df["negative_score"] = neg_scores

    pos_scores = []
    neg_scores = []
    neutral_scores = []
    for _title in tqdm(test_df["text"].values):
        _res = distilled_student_sentiment_classifier(_title, max_length=512)
        pos_scores.append(_res[0][0]["score"])
        neutral_scores.append(_res[0][1]["score"])
        neg_scores.append(_res[0][2]["score"])
    test_df["positive_score"] = pos_scores
    test_df["neutral_score"] = neutral_scores
    test_df["negative_score"] = neg_scores

    embeddings = cloudpickle.load(open("e5_large_embeddings.pkl", "rb"))

    # label encodingする
    oe = OrdinalEncoder()
    cloth_df[["Division Name", "Department Name", "Class Name"]] = oe.fit_transform(
        cloth_df[["Division Name", "Department Name", "Class Name"]]
    ).astype(int)
    # train testにマージ
    train_df = train_df.merge(cloth_df, how="left", on="Clothing ID")
    test_df = test_df.merge(cloth_df, how="left", on="Clothing ID")

    # 特徴量作成
    def get_kfold(train, n_splits, seed):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_series = []
        for fold, (idx_train, idx_valid) in enumerate(kf.split(train)):
            fold_series.append(pd.Series(fold, index=idx_valid))
        fold_series = pd.concat(fold_series).sort_index()
        return fold_series

    def get_targetencoding(train, test, folds: pd.Series, col: str, target_col):
        for fold in folds.unique():
            idx_train, idx_valid = (folds != fold), (folds == fold)
            group = train[idx_train].groupby(col)[target_col].mean().to_dict()
            train.loc[idx_valid, f"target_{col}"] = train.loc[idx_valid, col].map(group)
        group = train.groupby(col)[target_col].mean().to_dict()
        test[f"target_{col}"] = test[col].map(group)
        return train, test

    folds = get_kfold(train_df, 5, 42)
    cat_cols = ["Clothing ID", "Class Name"]
    for col in cat_cols:
        train_df, test_df = get_targetencoding(
            train_df, test_df, folds, col, target_col="Recommended IND"
        )

    # embeddingをマージ

    train_df = pd.concat(
        [
            train_df,
            pd.DataFrame(
                embeddings["train"],
                columns=[f"emb_{i}" for i in range(embeddings["train"].shape[1])],
            ),
        ],
        axis=1,
    )
    test_df = pd.concat(
        [
            test_df,
            pd.DataFrame(
                embeddings["test"],
                columns=[f"emb_{i}" for i in range(embeddings["test"].shape[1])],
            ),
        ],
        axis=1,
    )

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.01,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": 0.3,
        "lambda_l2": 0.3,
        "max_depth": 6,
        "num_leaves": 128,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "seed": 42,
    }
    except_cols = ["Review Text", "Title", "text", "Recommended IND", "Rating"]
    features = [col for col in train_df.columns if col not in except_cols]
    features = features + ["Cloth Rating Mean", "Cloth Rating Var"]

    # とりあえず StratifiedKFold で分割
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(train_df.shape[0])
    preds = np.zeros(test_df.shape[0])

    for fold_ix, (trn_, val_) in enumerate(
        skf.split(train_df, train_df["Recommended IND"])
    ):
        if fold_ix != 0:
            train_df = train_df.drop(["Cloth Rating Mean", "Cloth Rating Var"], axis=1)
            test_df = test_df.drop(["Cloth Rating Mean", "Cloth Rating Var"], axis=1)

        # fold ごとの train_df から Clothing ID に対する Rating Mean, Var を計算
        train_fold_df = train_df.iloc[trn_]
        cloth_df = train_fold_df.groupby("Clothing ID")[["Rating"]].agg(["mean", "var"])
        cloth_df.columns = ["Cloth Rating Mean", "Cloth Rating Var"]
        cloth_df = cloth_df.reset_index()
        cloth_df["Cloth Rating Mean"] = cloth_df["Cloth Rating Mean"].fillna(0)
        cloth_df["Cloth Rating Var"] = cloth_df["Cloth Rating Var"].fillna(0)

        train_df = train_df.merge(cloth_df, how="left", on="Clothing ID")
        test_df = test_df.merge(cloth_df, how="left", on="Clothing ID")

        trn_x = train_df.loc[trn_, features]
        trn_y = train_df.loc[trn_, "Recommended IND"]
        val_x = train_df.loc[val_, features]
        val_y = train_df.loc[val_, "Recommended IND"]

        trn_data = lgb.Dataset(trn_x, label=trn_y)
        val_data = lgb.Dataset(val_x, label=val_y)
        lgb_model = lgb.train(
            lgb_params,
            trn_data,
            valid_sets=[trn_data, val_data],
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)],
        )
        oof[val_] = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
        preds += (
            lgb_model.predict(test_df[features], num_iteration=lgb_model.best_iteration)
            / skf.n_splits
        )

    cv_score = roc_auc_score(train_df["Recommended IND"], oof)
    wandb.log({"AUC": cv_score})

    sub_df = pd.read_csv("data/sample_submission.csv")
    sub_df["target"] = preds
    sub_df.to_csv("predictions/pos_rating_lightgbm_13.csv", index=False)


if __name__ == "__main__":
    main()
