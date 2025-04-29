import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


class GestureDataset(Dataset):
    def __init__(self, feats, lbls):
        self.features = feats
        self.labels = lbls

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class GestureClassifier(nn.Module):
    def __init__(self, inputdim, numclasses):
        super().__init__()
        self.fc1 = nn.Linear(inputdim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, numclasses)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    csv_path = "./dataset/csv/hand_landmark_dataset.csv"
    model_save_path = "./models/hand_gesture_model.pth"
    os.makedirs("./dataset/csv", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val_encoded, y_test_encoded = train_test_split(
        X_test, y_test_encoded, test_size=0.5, random_state=42, stratify=y_test_encoded
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    batch_size = 32
    train_dataset = GestureDataset(X_train_tensor, y_train_tensor)
    val_dataset = GestureDataset(X_val_tensor, y_val_tensor)
    test_dataset = GestureDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_dim = X_train_scaled.shape[1]
    model_improved = GestureClassifier(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_improved.parameters(), lr=0.001)
    epochs = 100

    for epoch in range(epochs):
        model_improved.train()
        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model_improved(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
        model_improved.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in val_loader:
                outputs = model_improved(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # noqa
            print(
                f"Epoch [{epoch+1}/{epochs}], Validation Accuracy: {(100 * correct / total):.2f}%"
            )

    model_improved.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_predicted = []
        all_labels = []
        for features, labels in test_loader:
            outputs = model_improved(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # noqa
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        print(f"\nTest Accuracy: {(100 * correct / total):.2f}%")

        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix

        print("\nClassification Report:")
        print(
            classification_report(
                all_labels, all_predicted, target_names=label_encoder.classes_
            )
        )

        cm = confusion_matrix(all_labels, all_predicted)
        plt.figure(figsize=(12, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix (Improved Model)")
        plt.show()

    torch.save(model_improved.state_dict(), model_save_path)
    print(f"\nTrained improved model saved to: {model_save_path}")
