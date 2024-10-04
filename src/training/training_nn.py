import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast

from helpers.datapi import datas, url_data_call
from model.model import Early_Stopping, ModelCheckPoint, EarthquakeModel
from preprocessing.data_preprocessing import *
from helpers.utils import plot_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setting the device

#1 This function calls the data from the url and performs basic preprocessing.
def raw_data_prep(TimeSeries: bool) -> pd.DataFrame:
    """
    Calls and defines the data and returns a Pandas DataFrame with basic preprocssing.
    """
    df = pd.DataFrame()

    for key, values in datas.items():
        # print(f"{key} with value: {values}")
        pseudo_df = url_data_call(datas[key])

        df = pd.concat([df, pseudo_df])

    df = data_preprocessing(df, ts=TimeSeries) ## This function performs basic proecprocessing with an option of Timeseries or not.
    df = imput_encode(df) ## This function encodes and imputs the input data and fills the empty values.

    return df

# 2. This function performs feature engineering on the input dataframe.
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function performs feature engineering on the input dataframe.
    
    It performs the following operations:
    1. Creates event counts for different time windows
    2. Creates rolling window features
    """

    # Select columns with float data type
    float_cols = df.select_dtypes(include=['float64']).columns
    int32_cols = df.select_dtypes(include=['int32']).columns
    cat_cols = df.select_dtypes(include=['category']).columns

    # Fill NaN values in float columns with their respective means
    for col in float_cols:
        df[col] = df[col].fillna(df[col].mean())    

    for col in int32_cols:
        df[col] = df[col].astype('int64') # Converting the int32 columns to int64 [This inclues 'lat_lon_bin_encoded', 'magType']

    for col in cat_cols:
        df[col] = df[col].astype('float64') # Converting the category columns to float64 [This includes'lat_bin_mid', 'lon_bin_mid']

    # For Timeseries
    df.set_index(df['time'], inplace=True, drop=True)
    col_list= ['time_bin', 'time']
    for idx in col_list:
        if idx in df.columns:
            df = df.drop(columns=[idx])
    
    return df



# Feature Selection using Correlation Matrix
def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function performs feature selection on the input dataframe.
    
    It performs the following operations:
    1. Selects only numeric columns
    2. Calculates the correlation matrix
    3. Selects the top N features based on the correlation matrix
    """
        # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    corr_matrix = numeric_df.corr() # Calculate the Correaltion

        # Set a threshold for highly correlated features
    threshold = 0.8

    # Create an empty set to hold the columns to remove
    columns_to_drop = set()

    # Iterate over the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                columns_to_drop.add(colname)

    # Drop highly correlated features
    df_reduced = df.drop(columns=columns_to_drop)
    df_reduced = EnhancedCyclicTimeTransform(df_reduced) # Add Cyclic Time Features
    return df_reduced


# Defining Pararmeters
window_size = 15 * 24
target_column = ['mag', 'dmin', 'rms']

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Optimized dataset function call to avoid multiple reloads
def load_prep_dataset() -> pd.DataFrame:
    if 'cached_df' not in globals():
        print("Loading and preprocessing dataset for the first time...")
        global cached_df
        df = raw_data_prep(TimeSeries=False)
        df = event_counts_for_diff_window2(dataFrame=df)
        df = rolling_windows(new_df=df)
        df = feature_engineering(df)
        df = feature_selection(df)
        cached_df = df  # Cache the preprocessed dataframe
    return cached_df


# Defining the Target(s) and Variables
def VarTar(data) -> tuple:
    df = data

    # Separate features and targets
    X1 = df.drop(columns=target_column)
    Y1 = df[target_column]

    return X1, Y1


def scale_data(X1: pd.DataFrame, Y1: pd.DataFrame) -> tuple:
    """
    Scales the input features (X1) and targets (Y1) using standard scalers.
    
    Returns the scaled X and Y data along with the fitted scaler objects.
    
    Args:
        X1 (pd.DataFrame): Input feature data.
        Y1 (pd.DataFrame): Target data.

    Returns:
        scaled_X (np.array): Scaled input feature data.
        scaler_X (Scaler): Fitted scaler for input data.
        scaled_Y (np.array): Scaled target data.
        scaler_Y (Scaler): Fitted scaler for target data.
    """
    # Scaling the input features (X1)
    scaled_X, scaler_X = scaler_dataset(dataSet=X1)
    
    # Scaling the target variables (Y1)
    scaled_Y, scaler_Y = scaler_dataset(dataSet=Y1)
    
    return scaled_X, scaler_X, scaled_Y, scaler_Y


# Splitting data and caching it to avoid repeated splits
def split_data(data) -> tuple:
    """
    Splits the dataset into training, validation, and test sets.
    
    Calls the scale_data function to scale X and Y. Returns the splits
    along with the scalers for inverse transformations during testing.
    
    Returns:
        tuple: X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_Y
    """
    if 'cached_splits' not in globals():
        print("Splitting dataset for the first time...")
        global cached_splits

        # Separate features and targets
        X1 = VarTar(data)[0]
        Y1 = VarTar(data)[1]
        
        # Scale the features and targets
        scaled_X, scaler_X, scaled_Y, scaler_Y = scale_data(X1, Y1)

        # Creating the Single Step Multi Variable Seperate Sampler
        X, Y = SingleStepMultiVARS_SeperateSampler(scaled_X, scaled_Y, window_size, target_column)

        # Splitting the dataset into train, validation, and test sets
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        test_size = len(X) - train_size - val_size

        X_train, y_train = X[:train_size], Y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], Y[train_size+val_size:]

        # Cache the splits and scalers
        cached_splits = (X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_Y)
    
    return cached_splits


# Optimized DataLoader creation
def DataLoader_Conversion(data, test_data: bool = True) -> tuple:
    """
    Converts the training, validation, and test data splits into DataLoader objects.
    
    Returns the DataLoaders for each dataset and the fitted scalers for X and Y.
    
    Args:
        test_data (bool): Whether to return the test dataloader as well.
    
    Returns:
        tuple: train_dataloader, valid_dataloader, (test_dataloader), scaler_X, scaler_Y
    """
    cached_splits = split_data(data)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_Y = cached_splits

    train_tensor = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    valid_tensor = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_tensor = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_dataloader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_tensor, batch_size=BATCH_SIZE, shuffle=False)

    if test_data:
        test_dataloader = DataLoader(test_tensor, batch_size=BATCH_SIZE, shuffle=False)
        return train_dataloader, valid_dataloader, test_dataloader, scaler_X, scaler_Y
    return train_dataloader, valid_dataloader, scaler_X, scaler_Y
    # print(type(cached_splits))

# Training step for the Model
scaler = GradScaler()
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, early_stopping, checkpoint, experiment, logging=True ):
    
    if logging:
        with experiment.train():
            train_losses = []
            eval_losses = []

            for epoch in range(num_epochs):
                model.train()
                total_train_loss = 0
                train_predictions = []
                train_targets = []

                for i, (inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    
                    with autocast('cuda'):
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    # Backward pass and optimize
                    scaler.scale(loss).backward()

                    # # Gradient clipping
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    
                    scaler.step(optimizer)
                    scaler.update()

                    total_train_loss += loss.item()
                    train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    train_targets.extend(targets.cpu().numpy())

                    # Print loss every 2 iterations
                    if (i + 1) % 2 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                        experiment.log_metric("batch_loss", loss.item(), step=epoch * len(train_loader) + i)

                # Calculate average training loss and accuracy for the epoch
                avg_train_loss = total_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)


                # Evaluation loop
                model.eval()
                total_eval_loss = 0
                eval_predictions = []
                eval_targets = []

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        total_eval_loss += loss.item()
                        eval_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                        eval_targets.extend(targets.cpu().numpy())

                # Calculate average evaluation loss and accuracy for the epoch
                avg_eval_loss = total_eval_loss / len(val_loader)
                eval_losses.append(avg_eval_loss)


                print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}', f'Eval Loss: {avg_eval_loss:.4f}')

                # Log metrics
                experiment.log_metric("train_loss", avg_train_loss, epoch=epoch)
                experiment.log_metric("val_loss", avg_eval_loss, epoch=epoch)

                # Learning rate scheduling
                scheduler.step(avg_eval_loss)
                current_lr = scheduler.get_last_lr()[0]
                print(f"Learning Rate: {current_lr}")
                experiment.log_metric("learning_rate", current_lr, epoch=epoch)

                # Model checkpoint
                checkpoint(model, avg_eval_loss)

                # Early stopping
                early_stopping(avg_eval_loss)
                # if early_stopping.early_stop:
                #     print(f'Early stopping triggered after {epoch + 1} epochs')
                #     break
            return train_losses, eval_losses


# Test Step
def test_step(model, model_pth, scaler_Y):
    """
    Loads the model and performs inference on the test set.
    
    Args:
        model_pth (str): Path to the saved model.
        scaler_Y (Scaler): Fitted scaler for inverse transforming target predictions.
    """
    # Load the saved model
    model_path = model_pth
    loaded_model = model
    
    # Load the state dict
    loaded_model.load_state_dict(torch.load(model_path))
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
    experiment.log_metric("test_loss", test_loss)

    # Convert predictions and actuals to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Inverse transform predictions and actuals using the fitted scaler_Y
    predictions_original = scaler_Y.inverse_transform(predictions)
    actuals_original = scaler_Y.inverse_transform(actuals)

    # Calculate RMSE for each target variable
    for i, col in enumerate(target_column):
        rmse = np.sqrt(np.mean((predictions_original[:, i] - actuals_original[:, i])**2))
        print(f"RMSE for {col}: {rmse:.4f}")
        experiment.log_metric(f"RMSE_{col}", rmse)

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
        experiment.log_figure(figure_name=f"Actual_vs_Predicted_{col}", figure=fig)
        plt.close(fig)



# Calling the script 
if __name__ == "__main__":
    # DataLoader creation (with scalers for later use)
    train_dataloader, valid_dataloader, test_dataloader, scaler_X, scaler_Y = DataLoader_Conversion(load_prep_dataset())

    # Deining and logging HyperParameters
    hyper_params = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "window_size": window_size,
        "target_columns": target_column,
        "input_size": cached_splits[0].shape[-1],  # Determined after data is loaded
        "hidden_size": 64,
        "num_layers": 3,
        "output_size": len(target_column),
        "dropout_prob": 0.4,
        "weight_decay": 1e-5,
        "scheduler_patience": 10,
        "scheduler_factor": 0.3
    }

    # Initialize Comet ML experiment
    experiment = Experiment(
        api_key="WzhQCnsnCgodHTTrGLeFORixh",
        project_name="earthquake-preds",
        workspace="vintagep"
    )
    
    # Log hyperparameters to Comet ML
    experiment.log_parameters(hyper_params)
    
    

    # Model initialization using hyperparameters
    model = EarthquakeModel(
        input_size=hyper_params["input_size"], 
        hidden_size=hyper_params["hidden_size"], 
        num_layers=hyper_params["num_layers"], 
        output_size=hyper_params["output_size"], 
        dropout_prob=hyper_params["dropout_prob"]
    ).to(device)

    # Optimizer initialization
    optimizer = optim.Adam(
        model.parameters(), 
        lr=hyper_params["learning_rate"], 
        weight_decay=hyper_params["weight_decay"]
    )

    # Scheduler initialization
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=hyper_params["scheduler_factor"], 
        patience=hyper_params["scheduler_patience"]
    )


    criterion = nn.HuberLoss()

    # Callbacks
    model_checkpoint = ModelCheckPoint(file_path=r'C:\Projs\COde\Earthquake\eq_prediction\src\model\earthquake_best_model2.pth', verbose=True)
    early_stopping = Early_Stopping(patience=20, verbose=True)

    # Training loop
    with experiment.train():
        train_losses, val_losses = train_model(
            model, train_dataloader, valid_dataloader, 
            criterion, optimizer, scheduler, EPOCHS, 
            early_stopping, model_checkpoint
        )

    # Testing phase
    with experiment.test():
        test_step(model, model_pth=r'C:\Projs\COde\Earthquake\eq_prediction\src\model\earthquake_best_model2.pth', scaler_Y=scaler_Y)

    # Log final metrics and plots
    log_model(experiment, model, model_name="earthquake_model3")

    # Plot and log the loss curves
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    experiment.log_figure(figure_name="Loss Curves", figure=fig)
    plt.close(fig)

    # Log final metrics
    experiment.log_metric("final_train_loss", train_losses[-1])
    experiment.log_metric("final_val_loss", val_losses[-1])
    experiment.log_metric("best_val_loss", min(val_losses))

    # End the experiment
    experiment.end()

