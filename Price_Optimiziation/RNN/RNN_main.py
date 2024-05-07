import time
import pandas as pd
import numpy as np
from collections import deque
import random
from datetime import timedelta
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from Price_Optimization.RNN.libr.NeuralNet_functions import PriceLSTM, train_model, test_model, train_and_test


df = pd.read_csv('retail_price.csv')
df.sort_values(by=['product_id'], inplace=True)
#Define x and y variables
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

# Define your LSTM model
class PriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Output size should be 1 for binary classification

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the input
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Get output from the last time step
        return out.squeeze()  # Squeeze the output to match the shape of y_train_tensor

# Instantiate the model
input_size = X_train_tensor.shape[1]
hidden_size = 64
num_layers = 3  
output_size = 1
model = PriceLSTM(input_size, hidden_size, num_layers, output_size)
sequence_length = 1
# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert data to DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

num_epochs = 10000
# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Iterate over each unique product_id
    for product_id in X_train['product_id'].unique():
        # Filter data for the current product_id
        mask = X_train['product_id'] == product_id
        X_train_product = X_train_tensor[np.array(mask)]
        y_train_product = y_train_tensor[np.array(mask)]
        
        # Check if any data exists for the current product_id
        if len(X_train_product) > 0:
            # Perform training for the current product_id
            outputs = model(X_train_product.unsqueeze(1))
            outputs = outputs.squeeze()
            loss = criterion(outputs, y_train_product.squeeze())
            loss.backward()
            optimizer.step()

# Evaluation
with torch.no_grad():
    model.eval()
    for product_id in X_test['product_id'].unique():
        # Filter data for the current product_id
        mask = X_test['product_id'] == product_id
        X_test_product = X_test_tensor[np.array(mask)]
        y_test_product = y_test_tensor[np.array(mask)]

        y_pred_probs = model(X_test_product.unsqueeze(1))
        y_pred = (y_pred_probs > 0.5).float()

        # Calculate evaluation metrics for the current product_id
        mse = mean_squared_error(y_test_product, y_pred_probs)
        mae = mean_absolute_error(y_test_product, y_pred_probs)
        r2 = r2_score(y_test_product, y_pred_probs)

        print(f"Product ID: {product_id}")
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R-squared:", r2)

y_test_tensor = y_test_tensor.squeeze()
y_test_flat = y_test_tensor.squeeze().numpy().flatten()
y_pred_flat = y_pred.squeeze().numpy().flatten()
# Calculate Mean Squared Error
mse = mean_squared_error(y_test_tensor, y_pred_probs)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test_tensor, y_pred_probs)

# Calculate R-squared
r2 = r2_score(y_test_tensor, y_pred_probs)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

torch.save(model.state_dict(), 'price_lstm.pth')
#model = PriceLSTM(input_size, hidden_size, num_layers, output_size)
#model.load_state_dict(torch.load('price_lstm.pth'))