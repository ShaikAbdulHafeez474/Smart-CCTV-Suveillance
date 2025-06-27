import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VideoDataset
from model import CNN_LSTM
import torchvision.transforms as transforms
import optuna
import json
import os

# Dataset paths
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
SEQUENCE_LENGTH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform (no ToTensor as dataset.py already handles it)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def objective(trial):
    # Hyperparameters to tune
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    lr = trial.suggest_float("lr", 1e-5, 1e-3,log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)

    # Data loaders
    train_ds = VideoDataset(TRAIN_DIR, sequence_length=SEQUENCE_LENGTH, transform=transform)
    val_ds = VideoDataset(VAL_DIR, sequence_length=SEQUENCE_LENGTH, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = CNN_LSTM(hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train for 3 epochs only (fast search)
    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    val_acc = 100 * correct / total
    return val_acc

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

# Save best parameters
best_params = study.best_params
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("✅ Best Hyperparameters:")
print(best_params)
