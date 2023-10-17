from typing import Any, Dict, List

from xgboost import XGBRegressor
import xgboost as xgb

import pandas as pd
import numpy as np

# from comet_ml import Experiment
#
# experiment = Experiment(
#     api_key="WzhQCnsnCgodHTTrGLeFORixh",
#     project_name="general",
#     workspace="vintagep"
# )


def save_chkpt(model: Any):
    norm_mod = model.save_model('model/models2.json')
    return norm_mod


# Re-Training the model
def model_training(dat_X: pd.DataFrame, dat_y: pd.Series, test_X: pd.DataFrame) -> np.array:
    mod_inst = XGBRegressor(n_estimators=75, max_depth=3, eta=0.1, subsample=0.7)
    mod_inst.fit(dat_X, dat_y)

    preds = mod_inst.predict(test_X)

    return preds


def save_preds(preds: np.array, act_val: pd.Series) -> pd.DataFrame:
    data = {'Predicted_value': preds, 'Real_value': act_val}

    df2 = pd.DataFrame(data)
    df2.to_csv('saves/validation.csv')
    return df2
