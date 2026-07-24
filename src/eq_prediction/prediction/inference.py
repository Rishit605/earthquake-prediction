import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from eq_prediction.model import EarthquakeModel
from eq_prediction.training.training_nn import (
    target_column,
    load_prep_dataset,
    VarTar, scale_data, window_size
)

# Hyperparameters
input_size = 24
hidden_size = 64
num_layers = 3
output_size = len(target_column)
dropout_prob = 0.35

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setting the device
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "trained"
MODEL_PATH = MODEL_DIR / "earthquake_best_model.pth"

# Test Step
def test_step(loaded_model, test_dataloader, criterion, scaler_Y):
    loaded_model.eval()
    
    test_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = loaded_model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    test_loss /= len(test_dataloader)
    print(f"Test Loss: {test_loss:.4f}")
    # experiment.log_metric("test_loss", test_loss)

    # Convert predictions and actuals to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Inverse transform predictions and actuals
    predictions_original = scaler_Y.inverse_transform(predictions)
    actuals_original = scaler_Y.inverse_transform(actuals)

    # Calculate RMSE for each target variable
    for i, col in enumerate(target_column):
        rmse = np.sqrt(np.mean((predictions_original[:, i] - actuals_original[:, i])**2))
        print(f"RMSE for {col}: {rmse:.4f}")
        # experiment.log_metric(f"RMSE_{col}", rmse)

    # Log predictions vs actuals plot
    for i, col in enumerate(target_column):
        fig, ax = plt.subplots()
        ax.scatter(actuals_original[:, i], predictions_original[:, i], alpha=0.5)
        ax.plot([actuals_original[:, i].min(), actuals_original[:, i].max()], 
                [actuals_original[:, i].min(), actuals_original[:, i].max()], 
                'r--', lw=2)
        ax.set_xlabel(f'Actual {col}')
        ax.set_ylabel(f'Predicted {col}')
        ax.set_title(f'Actual vs Predicted {col}')
        # experiment.log_figure(figure_name=f"Actual_vs_Predicted_{col}", figure=fig)
        plt.close(fig)


def load_model(feature_count: int):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    model = EarthquakeModel(
        feature_count,
        hidden_size,
        num_layers,
        output_size,
        dropout_prob=dropout_prob,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

# Future forecasting
def future_forecast(model, last_sequence, scaler_X, scaler_Y, num_days, target_columns):
    model.eval()
    current_sequence = last_sequence.copy()
    forecasts = [] 
    with torch.no_grad():
        for _ in range(int(num_days * 24)):
            inputs = torch.as_tensor(current_sequence, dtype=torch.float32, device=device).unsqueeze(0)
            output = model(inputs)
            forecasts.append(output.cpu().numpy()[0])
            
            # Update the sequence for next prediction
            new_input = scaler_X.inverse_transform(current_sequence[-1].reshape(1, -1))
            new_input[:, :len(target_columns)] = scaler_Y.inverse_transform(output.cpu().numpy())
            new_input = scaler_X.transform(new_input)
            current_sequence = np.vstack((current_sequence[1:], new_input))
    
    return scaler_Y.inverse_transform(np.array(forecasts))


def generateDateRange(num_days, X1):
    last_date = pd.to_datetime(X1.index[-1])
    return pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=(int(num_days * 24)), freq='h')
     

def generate_future_predictions(data: bool, num_days=2):
    X1, Y1 = VarTar(load_prep_dataset(training=data))
    if len(X1) < window_size:
        raise ValueError(f"At least {window_size} rows are required for forecasting.")

    last_sequence = X1.iloc[-window_size:]
    _, scaler_X, _, scaler_Y = scale_data(X1, Y1)
    model = load_model(X1.shape[1])
    future_predictions = future_forecast(
        model,
        last_sequence.to_numpy(),
        scaler_X,
        scaler_Y,
        num_days,
        target_column,
    )
    future_dates = generateDateRange(num_days, X1)
    future_df = pd.DataFrame(future_predictions, columns=target_column, index=future_dates)
    return future_df, future_dates


if __name__ =='__main__':
    preds = generate_future_predictions(data=False, num_days = 1)
    print(preds)
