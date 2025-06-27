import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2, dropout=0.3):
        super(CNN_LSTM, self).__init__()

        # Load pretrained ResNet18 and remove the classifier head
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # remove final FC layer
        self.cnn = nn.Sequential(*modules)  # output: [B, 512, 1, 1]

        # LSTM with dropout (only works if num_layers > 1)
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=2,
                            batch_first=True, dropout=dropout)

        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()  # [batch, time, channel, height, width]
        x = x.view(B * T, C, H, W)

        cnn_features = self.cnn(x).view(B, T, -1)  # [B, T, 512]

        lstm_out, _ = self.lstm(cnn_features)  # [B, T, hidden_dim]
        last_hidden = lstm_out[:, -1, :]       # take output of last time step
        out = self.classifier(last_hidden)     # [B, num_classes]
        return out
