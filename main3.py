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

REGRESSION_DATA_PATH = PROJECT_ROOT / "data" / "engineered_data" / "FinalRegressionData.csv"


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("R2 score is undefined for empty inputs.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("R2 score inputs must not contain NaN or infinite values.")

    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - y_true.mean()))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


# Data Loading
class DataFile:
    def __init__(
        self,
        raw: bool = False,
        base: bool = False,
        regression: bool = True,
        save_data_flag: bool = True,
        data = None
        ) -> None:

        self._dataloader = None if data is not None else EQDataLoader(Saved=save_data_flag)
        self._save_data_flag = save_data_flag if self._dataloader is None else self._dataloader.save_data_flag
        self._raw = raw
        self._base = base
        self.z_scorer = None
        self._data = data
        self.COLS = ['dmin', 'dmin_km', 'elevation', 'gap', 'latitude', 'longitude', 'rms']
        self.INDEX_COLS = ['idx', 'old_idx', 'Unnamed: 0']


    def _load_final_data(
        self,
        cols=None,
        log_=True,
        sq=False,
        valid=False,
        test=False,
        transform=False
        ):

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
        ):

        df = self._dataloader.extra_data_prep() if self._data is None else self._data
        df = self._encoding(df)
        df = df.drop(columns=self.INDEX_COLS, errors="ignore")
        
        X_set, y_set = Data_Sets.split_dataset_xy(df, validation_flag=valid, test_flag=test)

        # df = DataPreprocessor(X_set).data_transform(log_t=log_, square=sq)
        df_experimental = ExperimentalDataPreprocessor(X_set).data_transform(square=sq)

        df = self._z_score_fit(df_experimental, self.COLS, transform=valid or test)
        
        df = df.reset_index(drop=True)

        return df, y_set

    def _load_all_final_data_mod(self, sq=False):
        df = self._dataloader.extra_data_prep() if self._data is None else self._data
        df = self._encoding(df)
        df = df.drop(columns=self.INDEX_COLS, errors="ignore")

        train_df, valid_df, test_df = Data_Sets.split_dataset(df)

        def split_xy(split_df):
            X_set = split_df.drop(columns=["mag"])
            y_set = split_df["mag"]
            return X_set, y_set

        train_X, train_y = split_xy(train_df)
        valid_X, valid_y = split_xy(valid_df)
        test_X, test_y = split_xy(test_df)

        train_X = ExperimentalDataPreprocessor(train_X).data_transform(square=sq)
        train_X = self._z_score_fit(train_X, self.COLS, transform=False)
        train_X = train_X.reset_index(drop=True)

        valid_X = ExperimentalDataPreprocessor(valid_X).data_transform(square=sq)
        valid_X = self._z_score_fit(valid_X, self.COLS, transform=True)
        valid_X = valid_X.reset_index(drop=True)

        test_X = ExperimentalDataPreprocessor(test_X).data_transform(square=sq)
        test_X = self._z_score_fit(test_X, self.COLS, transform=True)
        test_X = test_X.reset_index(drop=True)

        return {
            "X_train": train_X,
            "y_train": train_y,
            "X_valid": valid_X,
            "y_valid": valid_y,
            "X_test": test_X,
            "y_test": test_y,
        }


    def _z_score_fit(
        self,
        data,
        cols,
        transform=False
        ):
        if transform:
            if self.z_scorer is None:
                raise RuntimeError("Fit the training scaler before transforming validation/test data.")
            return self.z_scorer.transform_standard_Z(data.copy())

        self.z_scorer = ZScoreStandard(data.copy(), cols)
        return self.z_scorer.fit_standard_Z()

    def _load_raw_data(self) -> pd.DataFrame:
        # Loads the raw data with minimal refinement.
        df = self._dataloader.refine_og_data()
        return df

    def _load_base_data(self) ->  pd.DataFrame:
        return self._dataloader.data_prep()

    def _encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        # Encodes magType using DataEncoder's patcher.
        dE = DataEncoder(data)
        final_df = dE.patcher()
        if 'magType' in data.columns:
            final_df.drop('magType', axis=1, inplace=True)
        return final_df

    def __call__(self):
        if self._raw and self._data:
            return self._load_raw_data()
        elif self._base:
            return self._load_base_data()
        else:
            return self._load_final_data()



class DataExplorer:
    def __init__(self, reg_impute=False):
        self.base_data = None

        if reg_impute:
            from src.preprocessing.data_imputation_model import run_test_funcs
            self.base_data = run_test_funcs()
        else:
            pass

        self.fin_data = self._get_data()
        
    def _get_data(self):
        dLL = DataFile() if self.base_data is None else DataFile(data=self.base_data)

        return dLL._load_all_final_data_mod()

    
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
        return r2_score(y_true, y_pred)
    
    
    def train_and_evaluate(self, train_X, train_y, test_X, test_y):
        # reg = SGDRegressorScratch(train_X, train_y, 10000)
 
        self.fit(train_X, train_y)
        preds = self.predict(test_X)
        print("Predictions: ", preds)
        print("R2 Score: ", self._r2_loss(test_y, preds))
   

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

    def _call_model(self, max_depth=3, min_sample=2, valid=True, test=False):
        train_X, train_y = self.data_prep()
        pred_X, _ = self.data_prep(valid=valid, test=test)

        model = DecisionTreeR(max_depth=max_depth, min_sample=min_sample)
        model.fit(train_X, train_y)

        return model.predict(pred_X)


    def _evaluate_model(self, max_depth=3, min_sample=2, valid=True, test=False):
        if test:
            _, actuals = self.data_prep(test=True)
        elif valid:
            _, actuals = self.data_prep(valid=True)
        else:
            _, actuals = self.data_prep()

        preds = self._call_model(
            max_depth=max_depth,
            min_sample=min_sample,
            valid=valid,
            test=test,
        )
        return r2_score(actuals, preds)


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

        self.X_train, self.y_train = X_train, y_train

    def fit(self, X, y):
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("SGDRegressor must be fitted before predicting.")
        return self.model.predict(X)

    def _r2_loss(self, y_true, y_pred):
        return r2_score(y_true, y_pred)


    def train_and_evaluate(self, test_X, test_y):
        # reg = SGDRegressorScratch(train_X, train_y, 10000)
 
        self.fit(self.X_train, self.y_train)
        preds = self.predict(test_X)
        print("Predictions: ", preds)
        print("R2 Score: ", self._r2_loss(test_y, preds))


def summarize_missing_data(data):
    """Return a summary of invalid/missing values for a dataset (DataFrame/Series/ndarray).

    Summary columns: total_rows, null_count, pos_inf_count, neg_inf_count, non_finite_count, percent_null
    """
    # Normalize to DataFrame
    if isinstance(data, pd.Series):
        df = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        try:
            arr = np.asarray(data)
        except Exception:
            raise TypeError("Unsupported data type for summarize_missing_data")
        if arr.ndim == 0:
            raise TypeError(
                "summarize_missing_data expected a DataFrame, Series, or 1-D/2-D array; "
                f"got scalar value {data!r}."
            )
        if arr.ndim == 1:
            df = pd.DataFrame(arr, columns=["value"])
        elif arr.ndim == 2:
            df = pd.DataFrame(arr)
        else:
            raise TypeError(
                "summarize_missing_data expected a DataFrame, Series, or 1-D/2-D array; "
                f"got array with shape {arr.shape}."
            )

    total = len(df)
    summary = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        # For numeric-like values, check infinities and finiteness
        try:
            arr = series.values.astype(float)
            pos_inf = int(np.isposinf(arr).sum())
            neg_inf = int(np.isneginf(arr).sum())
            non_finite = int((~np.isfinite(arr)).sum())
        except Exception:
            pos_inf = neg_inf = non_finite = 0

        summary.append({
            "column": col,
            "total_rows": total,
            "null_count": null_count,
            "pos_inf_count": pos_inf,
            "neg_inf_count": neg_inf,
            "non_finite_count": non_finite,
            "percent_null": float(null_count) / total * 100 if total else 0.0,
        })

    return pd.DataFrame(summary).set_index("column")

if __name__ == "__main__":
    # explorer = DataExplorer()
    # d = explorer._get_data()
    # d = explorer.fin_data
    # d1 = DataExplorer().base_data
    # d2 = DataExplorer(exp=False).base_data
    # print(d['X_train'].shape)
    # print(d['X_train'].columns)

    ## Data Info
    # print(explorer._get_columns())
    # print(explorer.print_data_shapes())
    # print(explorer.print_skewness())
    # print(explorer.plot_histograms())

    # Linear Regressor Model : TODO: Models are performing worse than expected, issue unknown. Need to investigate further.
    # lr = SGDRegressorScratch(d['X_train'], d['y_train'], max_iter=50000).train_and_evaluate(d['X_test'], d['y_test'])
    # lr2 = LinearRegressor(d['X_train'], d['y_train'], max_iter=50000).train_and_evaluate(d['X_train'], d['y_train'], d['X_test'], d['y_test'])

    # Decision Tree Regressor
    dtr = DTRegressor()
    print("Predictions: ", dtr._call_model())
    print(dtr._evaluate_model(max_depth=11, min_sample=20)) # Best: depth=11, min_samples_split=20, R²=0.870592
    
    # d = EQDataLoader()
    # def info_print(data):
    #     print(data.head())
    #     print(data.shape)
    #     # print(data.sort_values(by='idx').index)
    #     print(data.isna().sum())
    #     print(data.tail())

    # info_print(d['X_train'])    
        
    # data = d.data_f
    # info_print(data)
    # print("\n")
    # data = d.refine_og_data()
    # info_print(data)
    # print()
