# backtest.py — Скрипт для гипотетической торговли

def backtest(model, data_loader, criterion, device):
    model.eval()
    total_profit = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Логика принятия решений на основе прогнозов
            if outputs > targets:  # Например, если предсказана рост
                total_profit += (outputs - targets).item()

    print(f"Итоговая прибыль на гипотетической торговле: {total_profit}")
