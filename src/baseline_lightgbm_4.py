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
from transformers import AutoModel, AutoTokenizer

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
    wandb.init(project="atmacup_17", name=f"lightgbm_4")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    cloth_df = pd.read_csv("data/clothing_master.csv")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    train_df["text"] = (
        "TITLE: "
        + train_df["Title"].fillna("none")
        + " [SEP] "
        + "Review Text: "
        + train_df["Review Text"].fillna("none")
    )
    test_df["text"] = (
        "TITLE: "
        + test_df["Title"].fillna("none")
        + " [SEP] "
        + "Review Text: "
        + test_df["Review Text"].fillna("none")
    )

    embeddings = {}
    for key, df in zip(["train", "test"], [train_df, test_df]):
        emb_list = []
        dataset = EmbDataset(df["text"].values, max_length=192)
        data_loader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=0,
            shuffle=False,
        )
        bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for iter_i, batch in bar:
            # input
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            with torch.no_grad():
                last_hidden_state, pooler_output, hidden_state = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True,
                    return_dict=False,
                )
                batch_embs = last_hidden_state.mean(dim=1)

            emb_list.append(batch_embs.detach().cpu().numpy())
        embeddings[key] = np.concatenate(emb_list)

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
        "learning_rate": 0.1,
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

    # とりあえず StratifiedKFold で分割
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(train_df.shape[0])
    preds = np.zeros(test_df.shape[0])

    for fold_ix, (trn_, val_) in enumerate(
        skf.split(train_df, train_df["Recommended IND"])
    ):

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
        oof[val_] = lgb_model.predict(val_x)
        preds += lgb_model.predict(test_df[features]) / skf.n_splits

    cv_score = roc_auc_score(train_df["Recommended IND"], oof)
    wandb.log({"AUC": cv_score})

    sub_df = pd.read_csv("data/sample_submission.csv")
    sub_df["target"] = preds
    sub_df.to_csv("predictions/baseline_lightgbm_4.csv", index=False)


if __name__ == "__main__":
    main()
