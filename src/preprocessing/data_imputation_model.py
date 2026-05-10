import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# print(PROJECT_ROOT)
# exit()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

from src.preprocessing.data_preprocessing import (
    EQDataLoader,
    DataPreprocessor,
    ZScoreStandard,
    Data_Sets,
    ExperimentalDataPreprocessor
)
from main_new import DataFile

# Functions
def get_enc(data):
    cp_df2 = data.copy()
    cp_df3 = DataFile()._encoding(cp_df2)
    return cp_df3

def prep_impute_data(data):
    x_t2 = get_enc(data)
    x_t2['dmin_missing'] = x_t2['dmin'].isna().astype(int)
    x_t2['gap_missing'] = x_t2['gap'].isna().astype(int)
    return x_t2


## KNN IMputer
class KNNImpute:
    def __init__(self, data, num_neighbors=5, cols=None ):
        self.k = num_neighbors
        self.cols = ['latitude', 'longitude', 'dmin', 'gap', 'rms'] if cols is None else cols
        self.df = data
        self.std_cache = None
        self.GEO_WEIGHT   = 1.0 # Geographic features get weight 1.0  → primary driver of neighbor selection
        self.OTHER_WEIGHT = 0.4 # Other features get weight 0.3 (default)       → present but subordinate
        self._impt = None


    def calc_std(self, df_imp):
        lat_std  = df_imp['latitude'].std()
        lon_std  = df_imp['longitude'].std()
        rms_std  = df_imp['rms'].std()
        gap_std  = df_imp['gap'].std()
        dmin_std = df_imp['dmin'].std()
        
        return {
            'lat_std': lat_std,
            'lon_std': lon_std,
            'rms_std': rms_std,
            'gap_std': gap_std,
            'dmin_std': dmin_std
        }

    def ensure_std(self, df_imp):
      """
      Compute std values once and reuse across methods.
      """
      if self.std_cache is None:
          self.std_cache = self.calc_std(df_imp)
      return self.std_cache

    def knn_impute_seismic(self, df):
        """
        KNN imputation for dmin and gap using geographically-weighted distance.
        """
        
        df_imp = df[self.cols].copy()

        # Reuse cached std values
        stds = self.ensure_std(df_imp)
        lat_std = stds['lat_std']
        lon_std = stds['lon_std']
        rms_std = stds['rms_std']
        gap_std = stds['gap_std']
        dmin_std = stds['dmin_std']

        df_scaled = pd.DataFrame()
        df_scaled['latitude']  = (df_imp['latitude']  / lat_std)  * self.GEO_WEIGHT
        df_scaled['longitude'] = (df_imp['longitude'] / lon_std)  * self.GEO_WEIGHT
        df_scaled['rms']       = (df_imp['rms']       / rms_std)  * self.OTHER_WEIGHT
        df_scaled['gap']       = (df_imp['gap']       / gap_std)  * self.OTHER_WEIGHT
        df_scaled['dmin']      = (df_imp['dmin']       / dmin_std) *self.OTHER_WEIGHT

        return df_scaled



    def _knn_fitputer(self, df, weights='distance'):

        k_imp_data = self.knn_impute_seismic(df)
    
        # ── Step 3: Run KNN on the scaled space ───────────────────────────────────
        imputer = KNNImputer(n_neighbors=self.k, weights=weights)
        self._impt = imputer

        imputed_scaled = imputer.fit_transform(k_imp_data)

        df_imputed_scaled = pd.DataFrame(imputed_scaled, columns=self.cols, index=k_imp_data.index)

        gap_std, dmin_std = self.std_cache['gap_std'], self.std_cache['dmin_std']

        # ── Step 4: Unscale back to original units ────────────────────────────────
        k_imp_data.loc[:, 'dmin'] = df_imputed_scaled['dmin'] * (dmin_std / self.OTHER_WEIGHT)
        k_imp_data.loc[:, 'gap']  = df_imputed_scaled['gap']  * (gap_std  / self.OTHER_WEIGHT)

        return k_imp_data


    def _knn_transputer(self, df, weights='distance'):

        k_imp_data = self.knn_impute_seismic(df)

        self.std_cache = self.ensure_std(self.df[self.cols].copy())  
        gap_std, dmin_std = self.std_cache['gap_std'], self.std_cache['dmin_std']
    
        # ── Step 3: Run KNN on the scaled space ───────────────────────────────────
        imputer = self._impt
        imputed_scaled = imputer.transform(k_imp_data)

        df_imputed_scaled = pd.DataFrame(imputed_scaled, columns=self.cols, index=k_imp_data.index)

        gap_std, dmin_std = self.std_cache['gap_std'], self.std_cache['dmin_std']

        # ── Step 4: Unscale back to original units ────────────────────────────────
        k_imp_data.loc[:, 'dmin'] = df_imputed_scaled['dmin'] * (dmin_std / self.OTHER_WEIGHT)
        k_imp_data.loc[:, 'gap']  = df_imputed_scaled['gap']  * (gap_std  / self.OTHER_WEIGHT)

        return k_imp_data


class C_KNN:
    def __init__(
        self,
        dataframe,
        st_d,
        weights='distance',
    
        
    ) -> None:
        self.df = dataframe
        self.weights = weights
        self.std = st_d
        self.knn_data = KNNImpute()
        self._impt = None
        self.cols = ['latitude', 'longitude', 'dmin', 'gap', 'rms']


    def _knn_fit(self):
        
        k_imp_data = self.knn_impute_seismic(self.df)

        # Obtain standard deviations using ensure_std from KNNImpute class
        self.std_cache = self.knn_data.ensure_std(self.df[self.cols].copy())
        gap_std, dmin_std = self.std_cache['gap_std'], self.std_cache['dmin_std']

        # Initialize the KNN imputer and fit/transform the data
        self._impt = self.knn_data.get_model()
        imputed_scaled = self._impt.fit_transform(k_imp_data)

        # Create a new DataFrame with imputed values, preserve index and columns
        df_imputed_scaled = pd.DataFrame(imputed_scaled, columns=self.cols, index=k_imp_data.index)

        # ── Step 4: Unscale back to original units ────────────────────────────────
        k_imp_data.loc[:, 'dmin'] = df_imputed_scaled['dmin'] * (dmin_std / self.OTHER_WEIGHT)
        k_imp_data.loc[:, 'gap']  = df_imputed_scaled['gap']  * (gap_std  / self.OTHER_WEIGHT)

        return k_imp_data


    def _knn_transform(self, data):
        """
        Transforms (imputes) the input data using the previously fitted KNN imputer (self._impt).
        """
        if self._impt is None:
            raise ValueError("KNN Imputer instance (self._impt) not yet fitted. Call _knn_fit() first.")
        
        # Prepare data for imputation (perserve index & columns)
        k_imp_data = self.knn_impute_seismic(data)
        imputed_scaled = self._impt.transform(k_imp_data)

        df_imputed_scaled = pd.DataFrame(imputed_scaled, columns=self.cols, index=k_imp_data.index)
        gap_std, dmin_std = self.std_cache['gap_std'], self.std_cache['dmin_std']

        # ── Step 4: Unscale back to original units ────────────────────────────────
        k_imp_data.loc[:, 'dmin'] = df_imputed_scaled['dmin'] * (dmin_std / self.OTHER_WEIGHT)
        k_imp_data.loc[:, 'gap']  = df_imputed_scaled['gap']  * (gap_std  / self.OTHER_WEIGHT)

        return k_imp_data


class ModelDataReCourse:
    """
    Handles recombination of imputed values and subsequent data processing
    in a modular and manageable manner.
    """

    def __init__(
        self,
        x_t,
        X_train_imp,
        X_valid_imp,
        Data_Sets,
        ExperimentalDataPreprocessor,
        get_enc,
        ZScoreStandard,
        missing_pred_count=2699
    ):
        """
        Args:
            x_t: Original dataframe to fill.
            X_train_imp: Imputed training DataFrame (expects 'gap' column).
            X_valid_imp: Imputed validation DataFrame (expects 'gap' column).
            Data_Sets: Utility/class for dataset split.
            ExperimentalDataPreprocessor: Preprocessor class (expects imput_data_transform).
            get_enc: Function for encoding.
            ZScoreStandard: ZScore normalization class.
            missing_pred_count: Number of prediction rows at the end of data (default: 2699).
        """
        self.x_t = x_t
        self.X_train_imp = X_train_imp
        self.X_valid_imp = X_valid_imp
        self.Data_Sets = Data_Sets
        self.ExperimentalDataPreprocessor = ExperimentalDataPreprocessor
        self.get_enc = get_enc
        self.ZScoreStandard = ZScoreStandard
        self.missing_pred_count = missing_pred_count

        # Outputs
        self.fin_combined_df2_s_X_e = None
        self.train_df2_X = None
        self.train_df2_y = None
        self.pred_df2_X = None
        self.pred_df2_y = None
        self.train_df_filled2_X = None
        self.pred_df_filled2_X = None
        self.cols_r = None

        self._run_pipeline()

    def _concat_gap(self):
        """
        Concatenates 'gap' columns from train and validation imputed dfs.
        """
        gap_df = pd.concat(
            [
                self.X_train_imp[['gap']],
                self.X_valid_imp[['gap']]
            ],
            axis=0
        )
        return gap_df

    def _patch_missing_gap(self, gap_df):
        """
        Patches missing 'gap' values in the original dataframe using gap_df.
        """
        df_filled = self.x_t.copy()
        df_filled['gap'] = df_filled['gap'].fillna(gap_df['gap'])
        return df_filled

    def _split_by_dmin(self, df):
        """
        Splits dataframe into training and prediction based on 'dmin' nan-ness.
        """
        train = df[df['dmin'].notnull()]
        pred = df[df['dmin'].isnull()]
        return train, pred

    def _recombine_and_sort(self, train, pred):
        """
        Concatenates and sorts dataframes by index.
        """
        combined_df = pd.concat([train, pred], axis=0).sort_index()
        return combined_df

    def _process_features(self, fin_combined_df):
        """
        Applies all preprocessing steps and splitting.
        """
        # Pull non-object columns (optional/for reference)
        _non_obj = fin_combined_df.select_dtypes(exclude=['object'])
        # Split into features/target
        X, y = self.Data_Sets.split_dataset_xy(fin_combined_df, target_col='dmin', set_split=False)
        # Select float columns only
        cols_r = X.select_dtypes(include=['float64']).columns
        # Preprocess imputations
        X_proc = self.ExperimentalDataPreprocessor(X).imput_data_transform(
            log_t=False, square=False, dmin_convert=False, cols=cols_r
        )
        # print(X_proc.columns)
        # exit()
        # Encode
        X_enc = self.get_enc(X_proc)
        return X_enc, y, cols_r

    def _final_split_and_scale(self, X_enc, y, cols_r):
        """
        Splits into train/predict sets and applies ZScore standardization.
        """
        split_idx = X_enc.shape[0] - self.missing_pred_count
        train_X, train_y = X_enc.iloc[:split_idx].copy(), y.iloc[:split_idx].copy()
        pred_X, pred_y = X_enc.iloc[split_idx:].copy(), y.iloc[split_idx:].copy()
        z_scorer = self.ZScoreStandard(train_X, cols_r)
        train_X_scaled = z_scorer.fit_standard_Z()
        pred_X_scaled = z_scorer.transform_standard_Z(pred_X)
        return train_X, train_y, pred_X, pred_y, train_X_scaled, pred_X_scaled

    def _run_pipeline(self):
        """
        Executes the full data preparation pipeline and stores results as attributes.
        """
        gap_df = self._concat_gap()
        df_filled = self._patch_missing_gap(gap_df)
        df_filled_train, df_filled_pred = self._split_by_dmin(df_filled)
        combined = self._recombine_and_sort(df_filled_train, df_filled_pred)
        combined_train, combined_pred = self._split_by_dmin(combined)
        fin_combined_df = pd.concat([combined_train, combined_pred], axis=0)

        X_enc, y, cols_r = self._process_features(fin_combined_df)
        self.fin_combined_df2_s_X_e = X_enc
        self.cols_r = cols_r

        self.train_df2_X, self.train_df2_y, self.pred_df2_X, self.pred_df2_y, self.train_df_filled2_X, self.pred_df_filled2_X = self._final_split_and_scale(X_enc, y, cols_r)


from sklearn.linear_model import LinearRegression

class SimpleLinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


if __name__== "__main__":
    # Load Data
    dL = EQDataLoader()
    # x_t = prep_impute_data(DataFile()._load_final_data())
    x_t = DataFile()._load_final_data()
    # Split data for imputer
    train_df, pred_df = Data_Sets.stratify_split(x_t)

    # KNNImputer
    # knn_imp = KNNImpute(data=train_df, num_neighbors=5, cols=['latitude', 'longitude', 'dmin', 'gap', 'rms'])
    knn_imp = C_KNN(dataframe=train_df, st_d=(), cols=['latitude', 'longitude', 'dmin', 'gap', 'rms']) ## TODO: Get the Stand deviation from KNN impute and run the custom model
    # knn_imp._knn_puter_f_t(train_df=train_df)
    X_train_imp = knn_imp._knn_fitputer(train_df)
    # # Transform validation/test with same fitted stats + imputer
    X_valid_imp = knn_imp._knn_transputer(pred_df)

    re_data = ModelDataReCourse(
        x_t=x_t,
        X_train_imp=X_train_imp,
        X_valid_imp=X_valid_imp,
        Data_Sets=Data_Sets,
        ExperimentalDataPreprocessor=ExperimentalDataPreprocessor,
        get_enc=get_enc,
        ZScoreStandard=ZScoreStandard
    )

    (
        train_df2_X, 
        train_df2_y, 
        pred_df2_X, 
        pred_df2_y, 
        train_df_filled2_X, 
        pred_df_filled2_X
    ) = (
        re_data.train_df2_X,
        re_data.train_df2_y,
        re_data.pred_df2_X,
        re_data.pred_df2_y,
        re_data.train_df_filled2_X,
        re_data.pred_df_filled2_X
    )

    model = SimpleLinearRegressionModel()
    model.fit(train_df_filled2_X, train_df2_y)
    print(model.predict(pred_df2_X))