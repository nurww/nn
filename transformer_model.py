# transformer_model.py

import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_encoder_layers, hidden_dim, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Linear(input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # src shape: (batch_size, seq_length, input_size)
        src = self.embedding(src)  # (batch_size, seq_length, hidden_dim)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # (seq_length, batch_size, hidden_dim) для Transformer
        memory = self.transformer_encoder(src)  # (seq_length, batch_size, hidden_dim)
        out = memory[-1, :, :]  # Последний временной шаг
        out = self.dropout(out)
        out = self.fc_out(out)  # (batch_size, output_size)
        return torch.sigmoid(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (seq_length, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
