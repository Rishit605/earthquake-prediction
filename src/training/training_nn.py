import os
import sys
import pandas as pd
import numpy as np

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import EarthquakeModel
from preprocessing.data_preprocessing import (
    data_preprocessing,
    imput_encode,
    var_and_tar,
    scaler_dataset,
    CyclicTimeTransform,
    SingleStepMultiVARS_SeperateSampler,
    split_dataset,
    )

from helpers.datapi import url, url_data_call

df = url_data_call(url)
df = data_preprocessing(df, ts=True)
df = imput_encode(df)
df = CyclicTimeTransform(df)

scaled_Dataset = scaler_dataset(df)
print(scaled_Dataset)

# df.to_csv("C:\Projs\COde\Earthquake\earthquake-prediction\earthquake_prediction\data\earthqukae_data.csv")

X, y = var_and_tar(df)

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
print(df)