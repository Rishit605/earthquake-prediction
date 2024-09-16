import os
import sys
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the Multivariate Time Series Forecasting Model
class WeatherForecastingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(WeatherForecastingModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.forecast_len = 7 * 24  # 7 days with 24 hours each
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = output[:, -1, :].unsqueeze(1)  # Get the last output and add a new dimension
        
        # Forecast for the next 7 days (forecast_len time steps)
        forecasts = []
        for _ in range(self.forecast_len):
            output, (h_n, c_n) = self.lstm(output, (h_n, c_n))
            forecast = self.fc(output[:, -1, :])
            forecasts.append(forecast)
            output = torch.cat([output, forecast.unsqueeze(1)], dim=1)  # Concatenate the forecast to the input
        
        forecasts = torch.cat(forecasts, dim=1)  # Combine all forecasts into a single tensor
        
        return forecasts


class EarthquakeModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.3):
        super(EarthquakeModel, self).__init__()
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class ModelCheckPoint:
    """
    Model checkpoint callback to save the best model based on validation loss.
    """

    def __init__(self, file_path='earthquake_best_model.pth', verbose=False):
        self.file_path = file_path
        self.verbose = verbose
        self.best_loss = float('inf')

    def __call__(self, model, valid_loss):
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            torch.save(model.state_dict(), self.file_path)

            if self.verbose:
                print(f"Saving New Beat Model with validation loss: {valid_loss:.4f}")  

class Early_Stopping:
    """  
    Early Stopping Callback function
    """

    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(f"Early Stopping Count er: {self.counter} out of {self.patience}")
            if self.patience >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0 



# # Example usage
# input_size = 5  # Number of input features (e.g., temperature, humidity, wind speed, etc.)
# hidden_size = 64
# output_size = 3  # Number of output features (e.g., temperature, precipitation, wind speed)
# num_layers = 2
# seq_len = 24  # Input sequence length of 24 hours
# num_epochs = 10

# # Train the model
# model = WeatherForecastingModel(input_size, hidden_size, output_size, num_layers)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(num_epochs):
#     for input_seq, target_seq in train_loader:
#         # Forward pass
#         output = model(input_seq)
#         loss = criterion(output, target_seq)
        
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     # Evaluate the model on the test set
#     model.eval()
#     with torch.no_grad():
#         test_loss = 0
#         for input_seq, target_seq in test_loader:
#             output = model(input_seq)
#             test_loss += criterion(output, target_seq).item()
#         test_loss /= len(test_loader)
    
#     print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}')