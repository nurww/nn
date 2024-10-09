# model_builder.py
import torch.nn as nn

# Построение модели на PyTorch
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

# Модель для расчета уровней поддержки и сопротивления
class SupportResistanceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(SupportResistanceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)
        self.fc_support = nn.Linear(hidden_size, output_size)  # Предсказание уровня поддержки
        self.fc_resistance = nn.Linear(hidden_size, output_size)  # Предсказание уровня сопротивления

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        support = self.fc_support(out)
        resistance = self.fc_resistance(out)
        return support, resistance