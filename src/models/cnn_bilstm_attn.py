# src/models/cnn_bilstm_attn.py
import torch.nn as nn
import torch

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim,1)
    def forward(self,x):
        # x: (batch, seq, dim)
        w = torch.softmax(self.proj(x).squeeze(-1), dim=1)
        return (x * w.unsqueeze(-1)).sum(dim=1)

class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self, n_features, seq_len, cnn_ch=64, lstm_hidden=128, num_classes=10):
        super().__init__()
        self.cnn = CNNBlock(n_features, cnn_ch)
        self.lstm = nn.LSTM(cnn_ch, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.att = Attention(lstm_hidden*2)
        self.fc = nn.Linear(lstm_hidden*2, num_classes)
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0,2,1)        # (batch, n_features, seq_len)
        x = self.cnn(x)             # (batch, cnn_ch, seq_len)
        x = x.permute(0,2,1)        # (batch, seq_len, cnn_ch)
        out, _ = self.lstm(x)       # (batch, seq_len, 2*lstm_hidden)
        att = self.att(out)         # (batch, 2*lstm_hidden)
        return self.fc(att)
