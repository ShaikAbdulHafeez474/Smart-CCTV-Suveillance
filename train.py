import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VideoDataset
from model import CNN_LSTM
import torchvision.transforms as transforms

# Fix fragmentation (optional but safe)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()  # Clear leftover cache if restarting

# Config (Manual Best Params)
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 9.6e-5
HIDDEN_DIM = 128
DROPOUT = 0.29
SEQUENCE_LENGTH = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# Transforms (Remove ToTensor since dataset.py already applies it)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset and loaders
train_ds = VideoDataset(TRAIN_DIR, sequence_length=SEQUENCE_LENGTH, transform=transform)
val_ds = VideoDataset(VAL_DIR, sequence_length=SEQUENCE_LENGTH, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, Optimizer
model = CNN_LSTM(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    train_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Validation
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
    print(f"           → Validation Accuracy: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("           ✔ Best model saved!")

print("✅ Training complete. Best validation accuracy:", best_val_acc)
