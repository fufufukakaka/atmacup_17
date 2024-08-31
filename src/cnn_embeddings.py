import cloudpickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define the CNN model
class EmbeddingCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmbeddingCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(
            256 * input_size // 8, 512
        )  # Adjust input size for fully connected layer
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Custom Dataset class
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.tensor(
                self.embeddings[idx], dtype=torch.float32
            ), torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return torch.tensor(self.embeddings[idx], dtype=torch.float32)


def main():
    # wandb.init(project="cnn_embeddings_classification_kfold")

    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    embeddings = cloudpickle.load(open("e5_large_embeddings.pkl", "rb"))

    # Prepare the dataset
    train_embeddings = embeddings["train"]
    test_embeddings = embeddings["test"]
    train_labels = train_df["Recommended IND"].values

    # Initialize K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_scores = []

    # Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_embeddings, train_labels)
    ):
        print(f"Fold {fold + 1}")

        # Create datasets for this fold
        train_dataset = Subset(
            EmbeddingDataset(train_embeddings, train_labels), train_idx
        )
        val_dataset = Subset(EmbeddingDataset(train_embeddings, train_labels), val_idx)
        test_dataset = EmbeddingDataset(test_embeddings)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Initialize model, loss function, and optimizer
        input_size = train_embeddings.shape[1]
        num_classes = 2  # Binary classification

        model = EmbeddingCNN(input_size, num_classes)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 10

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch_embeddings, batch_labels in train_loader:
                batch_embeddings, batch_labels = batch_embeddings.to(
                    device
                ), batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch_embeddings, batch_labels in val_loader:
                    batch_embeddings, batch_labels = batch_embeddings.to(
                        device
                    ), batch_labels.to(device)

                    outputs = model(batch_embeddings)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    val_preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_auc = roc_auc_score(val_labels, val_preds)

            # wandb.log(
            #     {"Train Loss": train_loss, "Val Loss": val_loss, "Val AUC": val_auc}
            # )

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
            )

        fold_score = roc_auc_score(val_labels, val_preds)
        fold_scores.append(fold_score)

        # Store out-of-fold predictions
        oof_preds[val_idx] = val_preds

        # Predict on the test set
        fold_test_preds = []

        with torch.no_grad():
            for batch_embeddings in test_loader:
                batch_embeddings = batch_embeddings.to(device)
                outputs = model(batch_embeddings)
                fold_test_preds.extend(
                    torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                )

        test_preds += np.array(fold_test_preds) / skf.n_splits

    # Calculate overall performance
    overall_auc = roc_auc_score(train_labels, oof_preds)
    print(f"Overall AUC: {overall_auc:.4f}")

    # wandb.log({"Overall AUC": overall_auc, "Fold AUC Scores": fold_scores})

    # Save the model
    torch.save(model.state_dict(), "cnn_embeddings_model_kfold.pth")

    # Save test predictions
    sub_df = pd.read_csv("data/sample_submission.csv")
    sub_df["Recommended IND"] = test_preds
    sub_df.to_csv("predictions/cnn_embeddings_test_preds.csv", index=False)


if __name__ == "__main__":
    main()
