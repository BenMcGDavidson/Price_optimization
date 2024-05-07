import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_test_convert(df):
    X = df[['product_id',
        'total_price',
        'freight_price',
        'product_photos_qty',
        'product_weight_g',
        'product_score',
        'customers',
        'weekday',
        'weekend',
        'holiday',
        'month',
        'year',
        'volume',
        'comp_1',
        'freight_price_comp1',
        'comp_2',
        'freight_price_comp2',
        'comp_3',
        'freight_price_comp3',
        'lag_price']]
    y = df[['unit_price']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    # Convert data to PyTorch tensors
    X_train_numeric = X_train.select_dtypes(include=[np.number])
    y_train_numeric = y_train.select_dtypes(include=[np.number])
    X_test_numeric = X_test.select_dtypes(include=[np.number])
    y_test_numeric = y_test.select_dtypes(include=[np.number])
    X_train_tensor = torch.tensor(X_train_numeric.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_numeric.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_numeric.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_numeric.values, dtype=torch.float32)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

class PriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

def test_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")

def create_data_loader(X, y, batch_size=32):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_and_test(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    test_model(model, test_loader)