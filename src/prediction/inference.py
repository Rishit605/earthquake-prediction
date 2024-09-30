import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from model.model import EarthquakeModel
from training.training_nn import (target_column, load_prep_dataset, VarTar, scale_data)

# Hyperparameters
input_size = 24
hidden_size = 64
num_layers = 3
output_size = len(target_column)
dropout_prob = 0.35

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setting the device

# Test Step
def test_step(loaded_model, model_pth):
    # # Load the saved model
    # model_path = model_pth
    
    # # Recreate the model architecture
    # loaded_model = EarthquakeModel(input_size, hidden_size, num_layers, output_size, dropout_prob=dropout_prob).to(device)
    
    # # Load the state dict
    # loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    
    test_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
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


# test_step(model_pth=r'C:\Projs\COde\Earthquake\eq_prediction\earthquake_best_model.pth')

def load_model():
    # Future Forecasts Generator
    model_path = r'C:\Projs\COde\Earthquake\eq_prediction\src\model\earthquake_best_model.pth'

    try:
        model = EarthquakeModel(input_size, hidden_size, num_layers, output_size, dropout_prob=dropout_prob).to(device)
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")

# Future forecasting
def future_forecast(model, last_sequence, scaler_X, scaler_Y, num_days, target_columns):
    model.eval()
    current_sequence = last_sequence.copy()
    forecasts = [] 
    with torch.no_grad():
        for _ in range(num_days * 24):
            inputs = torch.FloatTensor(current_sequence).unsqueeze(0).to("cuda")
            output = model(inputs)
            forecasts.append(output.cpu().numpy()[0])
            
            # Update the sequence for next prediction
            new_input = scaler_X.inverse_transform(current_sequence[-1].reshape(1, -1))
            new_input[:, :len(target_columns)] = scaler_Y.inverse_transform(output.cpu().numpy())
            new_input = scaler_X.transform(new_input)
            current_sequence = np.vstack((current_sequence[1:], new_input))
    
    return scaler_Y.inverse_transform(np.array(forecasts))


def generate_future_predictions(data: bool, num_days=2):
        num_days = num_days

        if data:
            # Loading the data for last sequence
            X1, Y1 = VarTar(load_prep_dataset())

            last_sequence = X1[-1:]
            model = load_model()

            scaler_X, scaler_Y = scale_data(X1, Y1)[1], scale_data(X1, Y1)[3]
        
            future_predictions = future_forecast(model, np.array(last_sequence), scaler_X, scaler_Y, num_days, target_column)
            last_date = pd.to_datetime(X1.index[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=(num_days * 24), freq='h')

            future_df = pd.DataFrame(future_predictions, columns=target_column, index=future_dates)
            # future_df.to_csv('eq_forecasts_after31122023.csv')
            return future_df
        else: 
            model = load_model()

            scaler_X, scaler_Y = scale_data(X1, Y1)[1], scale_data(X1, Y1)[3]
        
            future_predictions = future_forecast(model, np.array(last_sequence), scaler_X, scaler_Y, num_days, target_column)
            last_date = pd.to_datetime(X1.index[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=(num_days * 24), freq='h')

            future_df = pd.DataFrame(future_predictions, columns=target_column, index=future_dates)
            # future_df.to_csv('eq_forecasts_after31122023.csv')
            return future_df


if __name__ =='__main__':
    preds = generate_future_predictions(data=True)
    print(set(preds.columns))
    # print(future_df)
    # print(set(f'prediction_{i}' for i in range(1, 4)))