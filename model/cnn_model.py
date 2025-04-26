from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import os
import random
import torch.utils
import torch.utils.data

class CNNModel(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2, conv_layers=2, dropout=0.2, kernel_size=3, hidden_channels=128):
        super().__init__()
        layers = []
        in_channels = embed_dim
        for i in range(conv_layers):
            out_channels = hidden_channels
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            if i%4==3:
                layers.append(nn.AdaptiveAvgPool1d(1))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (batch, hidden_channels)
        x = self.dropout(x)
        x = self.fc(x)
        return x