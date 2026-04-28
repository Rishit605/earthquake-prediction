import os
import sys
from typing import List, Dict, final
from pathlib import Path

import pandas as pd
import numpy as np

import pandas as pd
from pathlib import Path

from sqlalchemy import false
# from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# from src.preprocessing.featur import FilterAndFill
from src.preprocessing.data_preprocessing import Data_Sets, EQDataLoader, DataPreprocessor, DataEncoder, ZScoreStandard
from src.helpers.utils import DataDist, plot_histograms, find_mean, find_variance
from src.preprocessing.feature_eng import FilterAndFill
from src.model.lr_scratch import LinearR


# Data Loading
class DataFile:
    def __init__(
        self,
        raw: bool = False,
        ) -> None:

        self._dataloader = EQDataLoader()
        self._raw = raw
        self.z_scorer = None

    def _load_final_data(
        self,
        log_=True,
        sq=False,
        valid=False,
        test=False,
        transform=False,
        ) -> pd.DataFrame:

        # Step 1: Get nicely prepped DataFrame from loader
        df = self._dataloader.data_prep()
        
        # Step 2: Split the Data (Default training set)
        X_set, y_set = Data_Sets.split_dataset_xy(df, validation_flag=valid, test_flag=test)

        # Step 3: Preprocess further (transforms)
        df = DataPreprocessor(X_set).data_transform(log_t=log_, sqaure=sq)

        # Step 4: Scale
        df = self._z_score_fit(df, transform=transform)

        # Step 5: Encode
        df = self._encoding(df)
        df = df.drop(columns=['others', 'idx']).copy()
        return df, y_set

    def _z_score_fit(
        self,
        data,
        transform=False
        ):
        
        self.z_scorer = ZScoreStandard(data)

        fitted = self.z_scorer.fit_standard_Z()

        if transform:
            return self.z_scorer.transform_standard_Z(data)
        else:
            return fitted


    def _load_raw_data(self) -> pd.DataFrame:
        # Loads the raw data with minimal refinement.
        df = self._dataloader.refine_og_data()
        return df

    def _encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        # Encodes magType using DataEncoder's patcher.
        # final_df = pd.DataFrame()
        dE = DataEncoder(data)
        final_df = dE.patcher()
        if 'magType' in data.columns:
            final_df.drop('magType', axis=1, inplace=True)
            return final_df

    def __call__(self) -> pd.DataFrame:
        if self._raw:
            # If asked for "raw" data, load the raw dataset and encode.
            df = self._load_raw_data()
            # df = self._encoding(df)
            # self._raw_encoded = True
            return df
        else:
            # Otherwise, get fully processed data pipeline.
            return self._load_final_data()


# Data Loading
# class DataFileNew:
#     def __init__(
#         self,
#         raw: bool = False
#         ) -> None:

#         self._dataloader = EQDataLoader()
#         self._raw = raw
#         self._raw_encoded = False

#     def data_transformation(self, data=None, transform="log", skew_threshold=0.5):
#         if data is None:
#             raise ValueError("data_transformation needs a DataFrame input.")

#         df = data.copy()
#         skew = DataDist.get_skew_for_all(df)

#         cols_to_transform = [
#             col for col, val in skew.items()
#             if pd.notnull(val) and abs(val) > skew_threshold and col in df.columns
#         ]

#         pre = DataPreprocessorNew(df)

#         if transform == "log":
#             return pre.log_transform_cols(cols_to_transform)
#         elif transform == "sqrt":
#             return pre.root_transform_cols(cols_to_transform, sq=True)
#         elif transform == "cbrt":
#             return pre.root_transform_cols(cols_to_transform, sq=False)
#         else:
#             return df

#     def _load_final_data(
#         self,
#         data=None,
#         target_col: str = "mag",
#         transform: str = "log",   # "log", "sqrt", "cbrt", or None
#         skew_threshold: float = 0.5
#         ):

#         # 1) load base data
#         df = self._dataloader.data_prep() if data is None else data.copy()

#         # 2) encode (if needed)
#         df = self._encoding(df) if "magType" in df.columns else df
#         # 3) split target first (prevents leakage)
#         y = df[target_col].copy()
#         X = df.drop(columns=[target_col]).copy()
#         # 4) transform only X based on skew
#         # X = self.data_transformation(
#         #     data=X,
#         #     transform=transform,
#         #     skew_threshold=skew_threshold
#         # )
#         # 5) scale only X
#         # X = DataPreprocessorNew(X).standard_Z()
#         return X, y

#     def _load_raw_data(self) -> pd.DataFrame:
#         # Loads the raw data with minimal refinement.
#         df = self._dataloader.refine_og_data()
#         return df

#     def _encoding(self, data: pd.DataFrame) -> pd.DataFrame:
#         # Encodes magType using DataEncoder's patcher.
#         # final_df = pd.DataFrame()
#         dE = DataEncoder(data)
#         final_df = dE.patcher()
#         if 'magType' in data.columns:
#             final_df.drop('magType', axis=1, inplace=True)
#             return final_df

#     def __call__(self) -> pd.DataFrame:
#         if self._raw:
#             # If asked for "raw" data, load the raw dataset and encode.
#             df = self._load_raw_data()
#             df = self._encoding(df)
#             self._raw_encoded = True
#             return df
#         else:
#             # Otherwise, get fully processed data pipeline.
#             return self._load_final_data()




if __name__ == "__main__":

#    # To Load Raw Data:
    # dL = DataFile(raw=True)
    # data = dL()
    # print(data.shape)

    # train, val, test = Data_Sets.split_dataset(data)
    # print(f"Training Data Shape: {train.shape}")
    # print(f"Valid Data Shape: {val.shape}")
    # print(f"Testing Data Shape: {test.shape}\n ")

    # print(f"Training Data Distribution: {DataDist.get_skew_for_all(train)}")
    # print(f"Training Data Distribution: {DataDist.get_skew_for_all(val)}")
    # print(f"Training Data Distribution: {DataDist.get_skew_for_all(test)} \n\n")

#    # To Load Training/validation/testing ready data
    dLL = DataFile()
    train_x, train_y = dLL._load_final_data()
    test_x, test_y = dLL._load_final_data(test=True, transform=True)
    print(f"Training Data Shape: {train_x.shape} //: Target Variable Shape: {train_y.shape}")
    print(f"Training Data Shape: {test_x.shape} //: Target Variable Shape: {test_y.shape}")

#    # Get Skewness of the data    
    print(DataDist.get_skew_for_all(train_x))
    print(DataDist.get_skew_for_all(test_x))

#    # Plot a histogram of the data
   # plot_histograms(train_x, train_x.columns)

#   # Train and test the model
    model = LinearR(train_x, train_y, 1000)
    # # print(model.matrix_conversion())
    print(model.fit(train_x, train_y))
    # print(model.predict(test_x, test_y))