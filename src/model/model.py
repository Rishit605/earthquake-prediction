import os
import sys
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Defining the Model Architecture.
class EarthquakeModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.3):
        super(EarthquakeModel, self).__init__()
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# Defining the Model Architecture.
class EarthquakeModel2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.3):
        super(EarthquakeModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Replace LSTM with GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout_prob, 
                          bidirectional=True)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state for GRU
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Use the output from the last time step
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
