from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import comet_ml
from comet_ml import Experiment
import os
from typing import List, Dict

import torch

from src.prediction.inference import future_forecast, generate_future_predictions
from src.model.model import EarthquakeModel
from src.prediction.inference import (
    input_size, 
    hidden_size,
    num_layers,
    output_size,
    dropout_prob
)

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setting the device

def load_model(model_path):
    try:
        model = EarthquakeModel(input_size, hidden_size, num_layers, output_size, dropout_prob=dropout_prob).to(device)
        model.load_state_dict(torch.load(model_path))
        return model, True
    except FileExistsError as e:
        print(f"Error loading model: {e}")
        return None, False


# # # Comet ML setup
# # comet_api_key = os.environ.get('COMET_API_KEY')
# # comet_project_name = "earthquake-prediction"
# # comet_workspace = "your-workspace-name"


class PredictionRequest(BaseModel):
    input_features: Dict[str, List[float]]

class PredictionResponse(BaseModel):
    predictions: Dict[str, List[float]]

@app.post("/train")
async def train_model():

    def load_dataset():
        try: 
            # Load and preprocess data
            data = pd.read_csv('new_data.csv')
            return data
        except:
            raise FileNotFoundError("Dataset Not Found") 

    if data.empty:
        print("Dataset is Empty!")
    else:
        X = data[['feature1', 'feature2', 'feature3']]  # Adjust features as needed
        y = data['target']
        
        X_scaled = scaler.fit_transform(X)
        
        # Initialize Comet experiment
        experiment = Experiment(
            api_key=comet_api_key,
            project_name=comet_project_name,
            workspace=comet_workspace,
        )
        
        # Train the model
        model.fit(X_scaled, y)
        
        # Log metrics to Comet
        experiment.log_metric("train_loss", model.loss_)
        experiment.log_model("earthquake_model", 'earthquake_model.joblib')
        
        # Save the updated model
        joblib.dump(model, 'earthquake_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        
        return {"message": "Model trained successfully"}


@app.post("/predict", response_model=PredictionResponse)
async def predict():
    try:    
        # Generate predictions using your custom function
        predictions_df = generate_future_predictions(data=True)

        # Ensure the output contains 3 prediction columns
        expected_outputs = set(predictions_df.columns)
        if set(predictions_df.columns) != expected_outputs:
            raise ValueError("Prediction output must contain 3 prediction columns")
        
        # Convert predictions DataFrame to dictionary
        predictions_dict = {col: predictions_df[col].tolist() for col in predictions_df.columns}
        
        return {"predictions": predictions_dict}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)