import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.functional as F

# -------- Modelo CNN para regresión angular --------
class SWECT_CNN_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((2, 3))

        self.fc1 = nn.Linear(64 * 8 * 30, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 64, 360]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(1)  # [B]

# -------- Dataset --------
class SWECTDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx)
        file = self.data.iloc[idx, 0]
        label = float(self.data.iloc[idx, 1])
        swect = torch.load(os.path.join(self.root_dir, file))  # [64, 360]
        return swect, label

# -------- Pérdida y métrica circular --------
class CircularRegressionLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred_rad = y_pred * np.pi / 180.0
        y_true_rad = y_true * np.pi / 180.0
        return torch.mean(1 - torch.cos(y_pred_rad - y_true_rad))

def circular_mae(y_pred, y_true):
    diff = torch.abs(y_pred - y_true) % 360
    return torch.mean(torch.minimum(diff, 360 - diff))

# -------- Entrenamiento y evaluación --------
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_mae = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        mae = circular_mae(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_mae += mae.item() * inputs.size(0)
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            mae = circular_mae(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_mae += mae.item() * inputs.size(0)
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)

# -------- Main --------
def main():
    csv_path = "/Users/salvaromero/Desktop/Gines/swect6/labels.csv"
    root_dir = "/Users/salvaromero/Desktop/Gines/swect6/swect_dataset"
    batch_size = 32
    epochs = 100
    subset_fraction = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SWECTDataset(csv_path, root_dir)
    total = len(dataset)
    subset_size = int(total * subset_fraction)
    indices = torch.randperm(total)[:subset_size]
    subset = Subset(dataset, indices)

    train_size = int(0.8 * subset_size)
    test_size = subset_size - train_size
    train_dataset, test_dataset = random_split(subset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = SWECT_CNN_Regression().to(device)
    criterion = CircularRegressionLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    patience = 7
    trigger_times = 0

    for epoch in range(epochs): 
        train_loss, train_mae = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "modelo_trigger-7.0-36k-32.pt")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("🔁 Early stopping triggered.")
                break

    torch.save(model.state_dict(), "modelo_7.0-36k-32.pt")
    print("✅ Modelo de regresión guardado como 'swect_cnn_regression_model.pt'")

# Para ejecutar:
if __name__ == "__main__":
    main()
