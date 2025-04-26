import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2, num_layers=2, 
                 dropout=0.2, hidden_channels=128, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_channels = hidden_channels
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_channels,
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0.0, 
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels * self.num_directions, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_channels * num_directions)
        out = out[:, -1, :]    # 取最后一个时间步
        out = self.dropout(out)
        out = self.fc(out)
        return out