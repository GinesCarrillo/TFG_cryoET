
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

# -------- CONFIG --------
SAVE_DIR = "/mnt/usb/data/maps"
LABELS_CSV = "/mnt/usb/data/labels.csv"
BATCH_SIZE = 16
EPOCHS = 20
block_size = 22
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Pérdida y métrica circular --------
def circular_loss(y_pred, y_true):
    y_pred_rad = y_pred * np.pi / 180.0
    y_true_rad = y_true * np.pi / 180.0
    return torch.mean(1 - torch.cos(y_pred_rad - y_true_rad))

def circular_mae(y_pred, y_true):
    diff = torch.abs(y_pred - y_true) % 360
    return torch.mean(torch.minimum(diff, 360 - diff))

# -------- Dataset --------
class AngleDatasetTorch(Dataset):
    def __init__(self, dataframe, data_dir):
        self.df = dataframe.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label = float(row['label'])
        arr = np.load(os.path.join(self.data_dir, filename))
        block = arr[0:block_size, 0:block_size, :]  # [22, 22, 360]
        max_val = np.max(block)
        if max_val > 0:
            block = block / max_val
        block = torch.tensor(block, dtype=torch.float32).unsqueeze(0)  # [1, 22, 22, 360]
        return block, torch.tensor(label, dtype=torch.float32)

# -------- Modelo --------
class CNN3DAngleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=2)
        self.pool2 = nn.MaxPool3d(2)
        self.dropout1 = nn.Dropout(0.6)

        # Calculamos automáticamente el tamaño tras convoluciones
        dummy = torch.zeros(1, 1, 22, 22, 360)
        x = self._forward_features(dummy)
        flat_dim = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flat_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.7)
        self.dropout3 = nn.Dropout(0.5)
        self.out = nn.Linear(128, 1)

    def _forward_features(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout1(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc3(x))
        return self.out(x).squeeze(1)

# -------- Entrenamiento --------
def train(model, dataloader, optimizer):
    model.train()
    total_loss, total_mae = 0, 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = circular_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_mae += circular_mae(y_pred, y).item() * x.size(0)
    return total_loss / len(dataloader.dataset), total_mae / len(dataloader.dataset)

# -------- Evaluación --------
def evaluate(model, dataloader):
    model.eval()
    total_loss, total_mae = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            loss = circular_loss(y_pred, y)
            total_loss += loss.item() * x.size(0)
            total_mae += circular_mae(y_pred, y).item() * x.size(0)
    return total_loss / len(dataloader.dataset), total_mae / len(dataloader.dataset)

# -------- Main --------
def main():
    df = pd.read_csv(LABELS_CSV)
    df = df.iloc[:100].copy()
    split_idx = int(0.8 * len(df))
    df_train = df[:split_idx]
    df_test = df[split_idx:]

    train_set = AngleDatasetTorch(df_train, SAVE_DIR)
    test_set = AngleDatasetTorch(df_test, SAVE_DIR)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = CNN3DAngleRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        train_loss, train_mae = train(model, train_loader, optimizer)
        test_loss, test_mae = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f}° | "
              f"Test Loss: {test_loss:.4f}, MAE: {test_mae:.2f}°")

    torch.save(model.state_dict(), "cnn3d_regresion_angular.pt")
    print("✅ Modelo guardado como 'cnn3d_regresion_angular.pt'")

# Descomenta para ejecutar:
if __name__ == "__main__":
    main()
