import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1,
)

import wandb

MODEL_ID = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


class EmbDataset(Dataset):
    def __init__(self, texts, max_length=192):
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, ix):
        token = self.tokenizer(
            self.texts[ix],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )
        return {
            "input_ids": torch.LongTensor(token["input_ids"]),
            "attention_mask": torch.LongTensor(token["attention_mask"]),
            "token_type_ids": torch.LongTensor(token["token_type_ids"]),
        }


def main():
    wandb.init(project="atmacup_17", name="lightgbm_16")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    cloth_df = pd.read_csv("data/clothing_master.csv")

    # ラベル平滑化(普通に)
    train_labels = train_df["Recommended IND"].values
    train_df["Recommended IND"] = train_df["Recommended IND"].replace(
        {1: 0.95, 0: 0.05}
    )

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

    # pos_scores = []
    # neg_scores = []
    # neutral_scores = []
    # for _title in tqdm(train_df["text"].values):
    #     _res = distilled_student_sentiment_classifier(_title, max_length=512)
    #     pos_scores.append(_res[0][0]["score"])
    #     neutral_scores.append(_res[0][1]["score"])
    #     neg_scores.append(_res[0][2]["score"])
    # train_df["positive_score"] = pos_scores
    # train_df["neutral_score"] = neutral_scores
    # train_df["negative_score"] = neg_scores

    # pos_scores = []
    # neg_scores = []
    # neutral_scores = []
    # for _title in tqdm(test_df["text"].values):
    #     _res = distilled_student_sentiment_classifier(_title, max_length=512)
    #     pos_scores.append(_res[0][0]["score"])
    #     neutral_scores.append(_res[0][1]["score"])
    #     neg_scores.append(_res[0][2]["score"])
    # test_df["positive_score"] = pos_scores
    # test_df["neutral_score"] = neutral_scores
    # test_df["negative_score"] = neg_scores

    embeddings = cloudpickle.load(open("e5_large_embeddings.pkl", "rb"))

    # label encodingする
    oe = OrdinalEncoder()
    cloth_df[["Division Name", "Department Name", "Class Name"]] = oe.fit_transform(
        cloth_df[["Division Name", "Department Name", "Class Name"]]
    ).astype(int)
    # train testにマージ
    train_df = train_df.merge(cloth_df, how="left", on="Clothing ID")
    test_df = test_df.merge(cloth_df, how="left", on="Clothing ID")
    # 特徴量作成はベースラインなのでスキップ
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
    except_cols = [
        "Review Text",
        "Title",
        "text",
        "Recommended IND",
        "Rating",
    ]
    features = [col for col in train_df.columns if col not in except_cols]

    # とりあえず StratifiedKFold で分割
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(train_df.shape[0])
    preds = np.zeros(test_df.shape[0])

    for fold_ix, (trn_, val_) in enumerate(
        skf.split(train_df, train_labels)
    ):

        trn_x = train_df.loc[trn_, features]
        trn_y = train_df.loc[trn_, "Recommended IND"]
        val_x = train_df.loc[val_, features]
        val_y = train_df.loc[val_, "Recommended IND"]

        trn_data = lgb.Dataset(trn_x, label=trn_y)
        val_data = lgb.Dataset(val_x, label=val_y)
        import pdb; pdb.set_trace()
        lgb_model = lgb.train(
            lgb_params,
            trn_data,
            valid_sets=[trn_data, val_data],
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)],
        )
        oof[val_] = lgb_model.predict(val_x)
        preds += lgb_model.predict(test_df[features]) / skf.n_splits

    cv_score = roc_auc_score(train_labels, oof)
    wandb.log({"AUC": cv_score})

    sub_df = pd.read_csv("data/sample_submission.csv")
    sub_df["target"] = preds
    sub_df.to_csv(
        "predictions/baseline_pos_label_smoothing_lightgbm_16.csv", index=False
    )


if __name__ == "__main__":
    main()
