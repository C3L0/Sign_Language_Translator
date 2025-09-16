import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# class HandPoseClassifier:
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         num_classes: int,
#         data: pd.DataFrame,
#         test_rate: float = 0.2,
#         y_label: str = "pose",
#     ):
#         self.num_classes = num_classes
#         self.data = data
#         self.test_rate = test_rate
#
#         self.model = self.MLP(input_dim, hidden_dim, num_classes)
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
#
#         self.X_train, self.X_test, self.y_train, self.y_test = self.StratifiedSplit(
#             self.data, self.test_rate
#         )
#
#     def StratifiedSplit(
#         self, df: pd.DataFrame, test_rate: float = 0.2, y_label: str = "pose"
#     ):
#         X = df.select_dtypes(include=[float, int]).values
#         y = df[y_label].values.astype(np.float32)
#
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_rate, stratify=y, random_state=42
#         )
#
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         X_test = torch.tensor(X_test, dtype=torch.float32)
#         y_train = torch.tensor(y_train, dtype=torch.long)
#         y_test = torch.tensor(y_test, dtype=torch.long)
#
#         return X_train, X_test, y_train, y_test
#
#     def Train(self, epochs: int = 10):  # add mini batches
#         for epoch in range(epochs):
#             self.optimizer.zero_grad()
#             outputs = self.model(self.X_train)
#             loss = self.criterion(outputs, self.y_train)
#             loss.backward()
#             self.optimizer.step()
#
#             print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}")
#
#     def Test(self):
#         with torch.no_grad():
#             outputs = self.model(self.X_test)
#             preds = torch.argmax(outputs, dim=1)
#             accuracy = (preds == self.y_test).float().mean()
#
#         return accuracy.item()
#
#     class MLP(nn.Module):
#         def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
#             super().__init__()
#             self.fc1 = nn.Linear(input_dim, hidden_dim)
#             self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#             self.fc3 = nn.Linear(hidden_dim, num_classes)
#
#             self.relu = nn.ReLU()
#
#         def forward(self, x):
#             x = self.relu(self.fc1(x))
#             x = self.relu(self.fc2(x))
#             x = self.fc3(x)
#
#             return x


class HandPoseClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        data: pd.DataFrame = None,
        test_rate: float = 0.2,
        y_label: str = "pose",
    ):
        self.num_classes = num_classes
        self.model = self.MLP(input_dim, hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Only build train/test if data is provided
        if data is not None:
            self.data = data
            self.test_rate = test_rate
            self.X_train, self.X_test, self.y_train, self.y_test = self.StratifiedSplit(
                self.data, self.test_rate, y_label
            )

    def StratifiedSplit(
        self, df: pd.DataFrame, test_rate: float = 0.2, y_label: str = "pose"
    ):
        X = (
            df.drop(columns=[y_label])
            .select_dtypes(include=[float, int])
            .values.astype(np.float32)
        )
        y = df[y_label].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_rate, stratify=y, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        return X_train, X_test, y_train, y_test

    def Train(self, epochs: int = 10):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = self.criterion(outputs, self.y_train)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

    def Test(self):
        with torch.no_grad():
            outputs = self.model(self.X_test)
            preds = torch.argmax(outputs, dim=1)
            accuracy = (preds == self.y_test).float().mean()
        return accuracy.item()

    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
