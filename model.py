# from typing import Any
# from xgboost import XGBRegressor
#
import pandas as pd

from data_pipline import *
from trainin_pipline import *

MODEL_PATH = 'model/models.json'


def training(path: str):
    PATH = path
    UNW_COLS = ['Origin Time', 'Location']
    TARGET_COL = ['Magnitude']
    TEST_SIZE = 0.2

    # Load the Data
    data = load_data(data_path=PATH, unwanted_cols=UNW_COLS)

    # Defining Features and Target
    inp_data = data.drop(columns=TARGET_COL)
    targ_data = data[TARGET_COL]

    # Splitting dataset
    train_test_dict = split_data(inputs=inp_data, outputs=targ_data, test_ratio=TEST_SIZE)

    # ReScaling the dataset
    train_test_dict['X_TRAIN'] = scaled(init_dat=train_test_dict['X_TRAIN'])
    train_test_dict['X_TEST'] = scaled(init_dat=train_test_dict['X_TEST'])

    # Model Training
    trained_model = model_training(dat_X=train_test_dict['X_TRAIN'], dat_y=train_test_dict['Y_TRAIN'],
                                   test_X=train_test_dict['X_TEST'])

    predictions = save_preds(preds=trained_model, act_val=train_test_dict['Y_TEST'])

    return predictions

"""
raise ValueError("Data must be 1-dimensional")
ValueError: Data must be 1-dimensional

TODO: RECTIFY THIS ERROR, WHILE USING 'save_preds' FUNCTION
"""

# # Load the model
# def load_model(mod_path: str):
#     model_isnt = XGBRegressor()
#
#     model = model_isnt.load_model(mod_path)
#     return model
#
#
# def predictions(modes: Any, ):
#     preds = modes.predict()


# X = data.drop(columns=['Magnitude'])
# y = data['Magnitude']

# print(split_data(inputs=scaled(X), outputs=y, test_ratio=0.2))

print(training('Official Website of National Center of Seismology.xlsx'))
# scaled_data = rescale_data(data, get_scaler(data))
# print(scaled_data)
