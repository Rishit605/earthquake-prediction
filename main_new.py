import sys
import warnings

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_preprocessing import Data_Sets, EQDataLoader, DataPreprocessor, DataEncoder, ZScoreStandard, ExperimentalDataPreprocessor
from src.helpers.utils import DataDist, plot_histograms
from src.model.lr_scratch import LinearR


# Data Loading
class DataFile:
    def __init__(
        self,
        raw: bool = False,
        base: bool = False,
        regression: bool = True,
        data = None
        ) -> None:

        self._dataloader = EQDataLoader()
        self._raw = raw
        self._base = base
        self.z_scorer = None
        self._data = data
        self.COLS = ['dmin', 'elevation', 'gap', 'latitude', 'longitude', 'rms']


    def _load_final_data(
        self,
        cols=None,
        log_=True,
        sq=False,
        valid=False,
        test=False,
        transform=False
        ) -> pd.DataFrame:

        df = self._dataloader.data_prep() if self._data is None else self._data
        X_set, y_set = Data_Sets.split_dataset_xy(df, validation_flag=valid, test_flag=test)

        df = DataPreprocessor(X_set).data_transform(log_t=log_, square=sq)
        df_experimental = ExperimentalDataPreprocessor(X_set).data_transform(log_t=log_, square=sq)

        df = self._z_score_fit(df_experimental, cols, transform=transform)
        df = self._encoding(df)
        df = df.reset_index(drop=True)

        return df, y_set


    def _load_final_data_mod(
        self,
        cols=None,
        sq=False,
        valid=False,
        test=False,
        transform=False
        ) -> pd.DataFrame:

        df = self._dataloader.data_prep() if self._data is None else self._data
        df = self._encoding(df)      
        
        X_set, y_set = Data_Sets.split_dataset_xy(df, validation_flag=valid, test_flag=test)

        # df = DataPreprocessor(X_set).data_transform(log_t=log_, square=sq)
        df_experimental = ExperimentalDataPreprocessor(X_set).data_transform(square=sq)

        df = self._z_score_fit(df_experimental, self.COLS, transform=transform)
        
        df = df.reset_index(drop=True)

        return df, y_set


    def _z_score_fit(
        self,
        data,
        cols,
        transform=False
        ):
        
        self.z_scorer = ZScoreStandard(data, cols)

        fitted = self.z_scorer.fit_standard_Z()

        if transform:
            return self.z_scorer.transform_standard_Z(data)
        else:
            return fitted

    def _load_raw_data(self) -> pd.DataFrame:
        # Loads the raw data with minimal refinement.
        df = self._dataloader.refine_og_data()
        return df

    def _load_base_data(self) ->  pd.DataFrame:
        return self._dataloader.data_prep()

    def _encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        # Encodes magType using DataEncoder's patcher.
        # final_df = pd.DataFrame()
        dE = DataEncoder(data)
        final_df = dE.patcher()
        if 'magType' in data.columns:
            final_df.drop('magType', axis=1, inplace=True)
            return final_df

    def __call__(self) -> pd.DataFrame:
        if self._raw and self._data:
            return self._load_raw_data()
        elif self._base:
            return self._load_base_data()
        else:
            return self._load_final_data()



class DataExplorer:
    def __init__(self):
        from src.preprocessing.data_imputation_model import run_test_funcs
        self.base_data = run_test_funcs()
        self.fin_data = self._get_data()
        # self.test_x, self.test_y = self.dLL._load_final_data(test=True, transform=True)

    def _get_data(self):
        dLL = DataFile(data=self.base_data)
        train_X, train_y = dLL._load_final_data_mod()
        valid_X, valid_y = dLL._load_final_data_mod(valid=True)
        test_X, test_y = dLL._load_final_data_mod(test=True)
        return {
            "X_train": train_X,
            "y_train": train_y,
            "X_valid": valid_X,
            "y_valid": valid_y,
            "X_test": test_X,
            "y_test": test_y,
        }

    
    def _get_columns(self, train=True, valid=False, test=False):
        # Print out the columns of both X and y sets of each dataset
        if train is not None:
            print("Train X Columns:")
            print(self.fin_data["X_train"].columns)
            print("Train y Columns:")
            print(self.fin_data["y_train"].name if hasattr(self.fin_data["y_train"], "name") else "y_train")
        elif valid:
            print("Validation X Columns:")
            print(self.fin_data["X_valid"].columns)
            print("Validation y Columns:")
            print(self.fin_data["y_valid"].name if hasattr(self.fin_data["y_valid"], "name") else "y_valid")
        elif test is not None:
            print("Test X Columns:")
            print(self.fin_data["X_test"].columns)
            print("Test y Columns:")
            print(self.fin_data["y_test"].name if hasattr(self.fin_data["y_test"], "name") else "y_test")
     


    def print_data_shapes(self, train=True, valid=False, test=False):
        if train is not None:
            print("Train X data shape:")
            print(self.fin_data["X_train"].shape)
            print("Train y data shape:")
            print(self.fin_data["y_train"].name if hasattr(self.fin_data["y_train"], "name") else "y_train")
        elif valid:
            print("Validation X data shape:")
            print(self.fin_data["X_valid"].shape)
            print("Validation y data shape:")
            print(self.fin_data["y_valid"].name if hasattr(self.fin_data["y_valid"], "name") else "y_valid")
        elif test is not None:
            print("Test X data shape:")
            print(self.fin_data["X_test"].shape)
            print("Test y data shape:")
            print(self.fin_data["y_test"].name if hasattr(self.fin_data["y_test"], "name") else "y_test")
     



    def print_skewness(self, train=True, valid=False, test=False):
        if train is not None:
            print("Skewness of Training Data:")
            # Only show skew for numeric columns
            numeric_cols = self.fin_data["X_train"].select_dtypes(include='number')
            print(numeric_cols.skew())
            print
        elif valid:
            print("Skewness of Validation Data:")
            numeric_cols = self.fin_data["X_valid"].select_dtypes(include='number')
            print(numeric_cols.skew())
        elif test is not None:
            print("Skewness of Test Data:")
            numeric_cols = self.fin_data["X_test"].select_dtypes(include='number')
            print(numeric_cols.skew())
 

    def plot_histograms(self, train=True, valid=False, test=False):
        
        if train is not None:
            print("Skewness of Training Data:")
            # Only show skew for numeric columns
            numeric_cols = self.fin_data["X_train"].select_dtypes(include='number')
            plot_histograms(numeric_cols, numeric_cols.columns)
        elif valid:
            print("Skewness of Validation Data:")
            numeric_cols = self.fin_data["X_valid"].select_dtypes(include='number')
            plot_histograms(numeric_cols, numeric_cols.columns)
        elif test is not None:
            print("Skewness of Test Data:")
            numeric_cols = self.fin_data["X_test"].select_dtypes(include='number')
            plot_histograms(numeric_cols, numeric_cols.columns)

        

class LinearRegressor:
    def __init__(self, X_train, y_train, max_iter=1000):
        self.model = LinearR(X_train, y_train, max_iter)
        self.fitted = False

    def fit(self, X, y):
        result = self.model.fit(X, y)
        self.fitted = True
        return result

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return self.model.predict(X)
   


def train_and_evaluate(train_X, train_y, test_X):
    reg = LinearRegressor(train_X, train_y, 1000)
    reg.fit(train_X, train_y)
    print("Predictions: ", reg.predict(test_X))


if __name__ == "__main__":
    explorer = DataExplorer()
    d = explorer.fin_data

    ## Data Info
    # print(explorer._get_columns())
    # print(explorer.print_data_shapes())
    # print(explorer.print_skewness())
    # print(explorer.plot_histograms())

    train_and_evaluate(d['X_train'], d['y_train'], d['X_test'])
