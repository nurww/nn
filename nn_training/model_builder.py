# model_builder.py

import torch.nn as nn
import torch

# Модель LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)  
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 2:  # Если нет временной оси (batch, input_size)
            x = x.unsqueeze(1)  # Добавляем временную ось (batch, 1, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Берем последний временной шаг
        return out

# Модель Transformer
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
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # (seq_length, batch_size, hidden_dim) для Transformer
        memory = self.transformer_encoder(src)
        out = memory[-1, :, :]  # Последний временной шаг
        out = self.dropout(out)
        out = self.fc_out(out)
        return torch.sigmoid(out)  # Или другая функция активации

# Позиционное кодирование для Transformer
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Функция для выбора модели
def build_model(model_type, input_size, output_size, **kwargs):
    if model_type == 'LSTM':
        return LSTMModel(input_size=input_size, hidden_size=kwargs['hidden_size'], num_layers=kwargs['num_layers'], output_size=output_size, dropout_rate=kwargs['dropout_rate'])
    elif model_type == 'Transformer':
        return TransformerModel(input_size=input_size, num_heads=kwargs['num_heads'], num_encoder_layers=kwargs['num_encoder_layers'], hidden_dim=kwargs['hidden_dim'], output_size=output_size, dropout=kwargs['dropout_rate'])
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
