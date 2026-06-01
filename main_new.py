import sys

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import SGDRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_preprocessing import Data_Sets, EQDataLoader, DataPreprocessor, DataEncoder, ZScoreStandard, ExperimentalDataPreprocessor
from src.helpers.utils import DataDist, plot_histograms
from src.model.lr_scratch import LinearR
from src.model.decision_tree_scratch import DecisionTreeR
from src.helpers.utils import r2_Loss


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
    def __init__(self, X_train, y_train, max_iter=10000):
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

    def _r2_loss(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Residual sum of squares (SSE)
        ss_res = np.sum((y_true - y_pred) ** 2)
        # Total sum of squares (SST)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        return 1 - (ss_res / ss_tot)
   

class DTRegressor:
    def __init__(self):        
        self.dT = DataExplorer()
        self.data = self.dT._get_data()
        self.data_flag = "train"

    def data_prep(self, valid=False, test=False):
        if valid:
            self.data_flag = "valid"
            return self.data['X_valid'].to_numpy(), self.data['y_valid'].to_numpy()
        elif test:
            self.data_flag = "test"
            return self.data['X_test'].to_numpy(), self.data['y_test'].to_numpy()
        else:
            return self.data['X_train'].to_numpy(), self.data['y_train'].to_numpy()  

    def _call_model(self, max_depth=3, min_sample=2):
        train_X, train_y = self.data_prep()
        valid_X, valid_y = self.data_prep(valid=True)

        model = DecisionTreeR(max_depth=max_depth, min_sample=min_sample)
        model.fit(train_X, train_y)

        return model.predict(valid_X)


    def _evaluate_model(self, max_depth=3, min_sample=2):
        if self.data_flag == "valid":
            _, actuals = self.data_prep(valid=True)
        elif self.data_flag == "test":
            _, actuals = self.data_prep(test=True)

        return r2_Loss(actuals, self._call_model(max_depth=max_depth, min_sample=min_sample))



class SGDRegressorScratch:
    def __init__(self, X_train=None, y_train=None, max_iter=1000, lr=0.01, tol=1e-5, random_state=None):
        from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
        
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.random_state = random_state
        self.fitted = False
        self.model = SklearnSGDRegressor(
            max_iter=max_iter,
            eta0=lr,
            learning_rate='constant',  # analogous to original; can be changed to 'invscaling', etc.
            tol=tol,
            random_state=random_state,
            penalty=None # No regularization for closest match to scratch
        )

        self.X_train, self.y_train = X_train. y_train

    def fit(self, X, y):
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("SGDRegressor must be fitted before predicting.")
        return self.model.predict(X)

    def _r2_loss(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # Residual sum of squares (SSE)
        ss_res = np.sum((y_true - y_pred) ** 2)
        # Total sum of squares (SST)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


    def train_and_evaluate(self, test_X, test_y):
        # reg = SGDRegressorScratch(train_X, train_y, 10000)

        self.fit(self.X_train, self.y_train)
        preds = self.predict(test_X)
        print("Predictions: ", preds)
        print("R2 Score: ", r2_Loss(test_y, preds))



if __name__ == "__main__":
    # explorer = DataExplorer()
    # d = explorer.fin_data
    # print(d['y_test'])

    ## Data Info
    # print(explorer._get_columns())
    # print(explorer.print_data_shapes())
    # print(explorer.print_skewness())
    # print(explorer.plot_histograms())

    # Linear Regressor Model
    # lr = SGDRegressorScratch(d['X_train'], d['y_train'], max_iter=50000).train_and_evaluate(d['X_test'], d['y_test'])

    # Decision Tree Regressor
    dtr = DTRegressor()
    print("Predictions: ", dtr._call_model())

    for depth in range(2, 21): 
        for sample in range(2, 21):
            print(f"Depth|Sample: {depth}|{sample} --> Model Score: {dtr._evaluate_model(max_depth=depth, min_sample=sample)}")
        
        

    # import matplotlib.pyplot as plt
    # t = np.linspace(10,90,20)
    # print(t)
    # print("\n", t[:-1])
    # print("\n", t[1:])
    # print(t[:-1] + t[1:])

    # plt.plot(t)
    # plt.show()