import torch
import numpy as np
from torch import nn
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

# -------- Clase para predicción individual --------
class SWECTPredictor:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SWECT_CNN_Regression().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, ect_input):
        """
        ect_input: np.ndarray or torch.Tensor de shape (64, 360)
        """
        if isinstance(ect_input, np.ndarray):
            ect_input = torch.tensor(ect_input, dtype=torch.float32)
        if ect_input.ndim != 2 or ect_input.shape != (64, 360):
            raise ValueError("La ECT debe tener shape (64, 360)")

        ect_input = ect_input.unsqueeze(0).to(self.device)  # [1, 64, 360]
        with torch.no_grad():
            pred = self.model(ect_input)
        angle = pred.item() % 360
        return angle
