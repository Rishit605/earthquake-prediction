from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from typing import List, Dict

import comet_ml
from comet_ml import Experiment
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dotenv import load_dotenv

from src.training.training_nn import (
    DataLoader_Conversion, load_prep_dataset,
    EPOCHS, LEARNING_RATE, BATCH_SIZE,
    target_column, window_size,
    train_model, test_step
)
from src.prediction.inference import (
    future_forecast,
    generate_future_predictions,
    generateDateRange,
    input_size, 
    hidden_size,
    num_layers,
    output_size,
    dropout_prob
)

load_dotenv()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str) -> tuple:
    try:
        model = EarthquakeModel(input_size, hidden_size, num_layers, output_size, dropout_prob=dropout_prob).to(device)
        model.load_state_dict(torch.load(model_path))
        return model, True
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return None, False

# Comet ML setup
comet_api_key = os.environ.get('API_KEY')
comet_project_name = os.environ.get('PROJECT_NAME')
comet_workspace = os.environ.get('WORKSPACE')

# Initialize Comet ML experiment
experiment = Experiment(
    api_key=comet_api_key,
    project_name=comet_project_name,
    workspace=comet_workspace,
)

# DataClasses
class PredictionRequest(BaseModel):
    input_features: Dict[str, List[float]]

class PredictionResponse(BaseModel):
    predictions: Dict[str, List[float]]
    dates: List[str]  # Added to include future dates

# Training Protocol
@app.post("/train")
async def training():
    def load_dataset() -> tuple:
        try:
            train_dataloader, valid_dataloader, test_dataloader, scaler_X, scaler_Y = DataLoader_Conversion(load_prep_dataset())
            return train_dataloader, valid_dataloader, test_dataloader, scaler_X, scaler_Y
        except FileNotFoundError:
            raise FileNotFoundError("Dataset Not Found")

    train_dataloader, valid_dataloader, test_dataloader, scaler_X, scaler_Y = load_dataset()

    if len(train_dataloader) == 0:
        print("Dataset is Empty!")
        return {"message": "Dataset is Empty!"}

    first_batch = next(iter(train_dataloader))
    inputs, targets = first_batch  # Adjust this based on your DataLoader's output structure

    # Defining and logging HyperParameters
    hyper_params = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "window_size": window_size,
        "target_columns": target_column,
        "input_size": inputs.shape[-1],
        "hidden_size": 64,
        "num_layers": 4,
        "output_size": len(target_column),
        "dropout_prob": 0.4,
        "weight_decay": 1e-5,
        "scheduler_patience": 10,
        "scheduler_factor": 0.3
    }

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
    model_checkpoint = ModelCheckPoint(file_path=r'C:\Projs\COde\Earthquake\eq_prediction\src\model\earthquake_best_model3.pth', verbose=True)
    early_stopping = Early_Stopping(patience=20, verbose=True)

    # Training loop
    with experiment.train():
        train_losses, val_losses = train_model(
            model, train_dataloader, valid_dataloader, 
            criterion, optimizer, scheduler, EPOCHS, 
            early_stopping, model_checkpoint, experiment
        )

    # Testing phase
    with experiment.test():
        test_step(model, model_pth=r'C:\Projs\COde\Earthquake\eq_prediction\src\model\earthquake_best_model2.pth', scaler_Y=scaler_Y)

    # Log final metrics
    experiment.log_metric("final_train_loss", train_losses[-1])
    experiment.log_metric("final_val_loss", val_losses[-1])
    experiment.log_metric("best_val_loss", min(val_losses))

    experiment.end()        
    return {"message": "Model trained successfully"}

# Inference Protocol
@app.post("/predict", response_model=PredictionResponse)
async def predict():
    try:
        # Generate predictions using your custom function
        predictions_df, future_dates = generate_future_predictions(data=True, num_days=0.5)

        # Ensure the output contains 3 prediction columns
        expected_outputs = set(predictions_df.columns)
        if set(predictions_df.columns) != expected_outputs:
            raise ValueError("Prediction output must contain 3 prediction columns")

        # Convert predictions DataFrame to dictionary
        predictions_dict = {col: predictions_df[col].tolist() for col in predictions_df.columns}

        # Convert future_dates from Timestamp to string format with microseconds
        future_dates_str = [date.strftime('%Y-%m-%d %H:%M:%S.%f') for date in future_dates]

        return {"predictions": predictions_dict, "dates": future_dates_str}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
