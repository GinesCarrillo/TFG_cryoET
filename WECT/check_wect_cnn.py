import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

# -------- Dataset --------
class SWECTDataset(Dataset):
    def __init__(self, data, root_dir):
        if isinstance(data, str):  # si se pasa ruta CSV
            self.data = pd.read_csv(data)
        else:  # si se pasa DataFrame directamente
            self.data = data.reset_index(drop=True)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[int(idx)]
        fname = row['filename']
        label = float(row['label'])
        swect = torch.load(os.path.join(self.root_dir, fname))  # [64, 360]
        return swect, label

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

# -------- Circular MAE --------
def circular_mae(y_pred, y_true):
    diff = torch.abs(y_pred - y_true) % 360
    return torch.mean(torch.minimum(diff, 360 - diff))

# -------- Evaluación --------
def evaluate_model(model_path, csv_path, root_dir, batch_size=64, num_samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Leer subset del CSV
    full_df = pd.read_csv(csv_path).iloc[:num_samples].reset_index(drop=True)
    dataset = SWECTDataset(full_df, root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Modelo
    model = SWECT_CNN_Regression().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = circular_mae(pred, y)
            total_loss += loss.item() * X.size(0)
            all_preds.append(pred.cpu().numpy()%360)
            all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / len(dataset)
    print(f"✅ MAE circular en test: {avg_loss:.2f}°")

    return np.concatenate(all_preds), np.concatenate(all_labels), full_df


# -------- MAIN --------
if __name__ == "__main__":
    csv_path = "/Users/salvaromero/Desktop/Gines/swect3/labels.csv"
    root_dir = "/Users/salvaromero/Desktop/Gines/swect3/swect_dataset"
    model_path = "modelo_trigger-2.0-36k-128.pt"  # o "best_model.pt"

    preds, labels, df_test = evaluate_model(model_path, csv_path, root_dir, num_samples=500)

    # Guardar CSV
    sample_ids = df_test['filename'].str.extract(r'sample_(\d+)')[0].astype(int)
    df_out = pd.DataFrame({
        "sample_id": sample_ids,
        "real_angle": labels,
        "predicted_angle": preds
    })
    output_csv = "predicciones_test.csv"
    df_out.to_csv(output_csv, index=False)
    print(f"📁 CSV guardado en: {output_csv}")

    # Opcional: gráfico
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.scatter(labels, preds, s=10, alpha=0.5)
    plt.xlabel("Ángulo real (label)")
    plt.ylabel("Ángulo predicho")
    plt.title("Predicción vs Real en test")
    plt.grid(True)
    plt.plot([0, 360], [0, 360], 'r--')
    plt.show()
