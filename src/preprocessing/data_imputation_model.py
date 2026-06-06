"""Imputation helpers, KNN-based gap/dmin imputation, and post-imputation dataset assembly."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from types import SimpleNamespace


from src.preprocessing.data_preprocessing import (
    EQDataLoader,
    DataEncoder,
    DataPreprocessor,
    ZScoreStandard,
    Data_Sets,
    ExperimentalDataPreprocessor,
)

# ------------------------------------------------------------------------------------------------- #

DEFAULT_IMPUTE_COLS = ["latitude", "longitude", "dmin", "gap", "rms"]


def get_enc(data):
    cp_df2 = data.copy()
    dE = DataEncoder(cp_df2)  # from data_preprocessing
    cp_df3 = dE.patcher()
    if 'magType' in cp_df2.columns:
        cp_df3 = cp_df3.drop('magType', axis=1)
    return cp_df3


def prep_impute_data(data):
    x_t2 = get_enc(data)
    x_t2["dmin_missing"] = x_t2["dmin"].isna().astype(int)
    x_t2["gap_missing"] = x_t2["gap"].isna().astype(int)
    return x_t2


def _unscale_dmin_gap(k_imp_data, df_imputed_scaled, std_cache, other_weight):
    """Map scaled KNN outputs back to original dmin/gap units (same formula as before)."""
    gap_std = std_cache["gap_std"]
    dmin_std = std_cache["dmin_std"]
    k_imp_data.loc[:, "dmin"] = df_imputed_scaled["dmin"] * (dmin_std / other_weight)
    k_imp_data.loc[:, "gap"] = df_imputed_scaled["gap"] * (gap_std / other_weight)


def resolve_numeric_impute_columns(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    ) -> list[str]:
    """Return columns that exist, are numeric, and are not boolean."""

    def _ok(series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

    if cols is None:
        want = [c for c in DEFAULT_IMPUTE_COLS if c in df.columns]
    else:
        want = [c for c in cols if c in df.columns]
    return [c for c in want if _ok(df[c])]



class ImputationStats:
    @staticmethod
    def calc_std(df_imp: pd.DataFrame, cols: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in cols:
            v = df_imp[c].std()
            if pd.isna(v) or v == 0:
                out[c] = 1.0
            else:
                out[c] = float(v)
        return out


class KNNImputeOLD:
    def __init__(self, data=None, num_neighbors=5, cols=None):
        self.k = num_neighbors
        self.cols = list(DEFAULT_IMPUTE_COLS) if cols is None else list(cols)
        self.df = data
        self.std_cache = None
        self.GEO_WEIGHT = 1.0
        self.OTHER_WEIGHT = 0.4
        self._impt = None

    def ensure_std(self, df_imp):
        """Compute std values once and reuse across methods."""
        if self.std_cache is None:
            self.std_cache = ImputationStats.calc_std(df_imp)
        return self.std_cache

    def knn_impute_seismic(self, df):
        """KNN imputation for dmin and gap using geographically-weighted distance."""
        df_imp = df[self.cols].copy()
        stds = self.ensure_std(df_imp)
        lat_std = stds["lat_std"]
        lon_std = stds["lon_std"]
        rms_std = stds["rms_std"]
        gap_std = stds["gap_std"]
        dmin_std = stds["dmin_std"]

        df_scaled = pd.DataFrame(
            {
                "latitude": (df_imp["latitude"] / lat_std) * self.GEO_WEIGHT,
                "longitude": (df_imp["longitude"] / lon_std) * self.GEO_WEIGHT,
                "rms": (df_imp["rms"] / rms_std) * self.OTHER_WEIGHT,
                "gap": (df_imp["gap"] / gap_std) * self.OTHER_WEIGHT,
                "dmin": (df_imp["dmin"] / dmin_std) * self.OTHER_WEIGHT,
            },
            index=df_imp.index,
        )
        return df_scaled

    def _knn_fitputer(self, df, weights="distance"):
        k_imp_data = self.knn_impute_seismic(df)
        imputer = KNNImputer(n_neighbors=self.k, weights=weights)
        self._impt = imputer
        imputed_scaled = imputer.fit_transform(k_imp_data)
        df_imputed_scaled = pd.DataFrame(
            imputed_scaled, columns=self.cols, index=k_imp_data.index
        )
        _unscale_dmin_gap(
            k_imp_data, df_imputed_scaled, self.std_cache, self.OTHER_WEIGHT
        )
        return k_imp_data

    def _knn_transputer(self, df, weights="distance"):
        k_imp_data = self.knn_impute_seismic(df)
        self.std_cache = self.ensure_std(self.df[self.cols].copy())
        imputer = self._impt
        imputed_scaled = imputer.transform(k_imp_data)
        df_imputed_scaled = pd.DataFrame(
            imputed_scaled, columns=self.cols, index=k_imp_data.index
        )
        _unscale_dmin_gap(
            k_imp_data, df_imputed_scaled, self.std_cache, self.OTHER_WEIGHT
        )
        return k_imp_data


class KNNImpute:
    SEISMIC_COLS = frozenset(["latitude", "longitude", "dmin", "gap", "rms"])

    def __init__(self, data=None, num_neighbors: int = 5, cols: list[str] | None = None):
        self.k = num_neighbors
        self.cols = list(DEFAULT_IMPUTE_COLS) if cols is None else list(cols)
        self.df = data
        self.std_cache: dict[str, float] | None = None
        self.GEO_WEIGHT = 1.0
        self.OTHER_WEIGHT = 0.4
        self._impt = None

    def _is_seismic_block(self) -> bool:
        return frozenset(self.cols) == self.SEISMIC_COLS

    def ensure_std(self, df_imp: pd.DataFrame) -> dict[str, float]:
        if self.std_cache is None:
            self.std_cache = ImputationStats.calc_std(df_imp, self.cols)
        return self.std_cache

    def _scale_frame(self, df_imp: pd.DataFrame, stds: dict[str, float]) -> pd.DataFrame:
        parts: dict[str, pd.Series] = {}
        if self._is_seismic_block():
            lat_s, lon_s = stds["latitude"], stds["longitude"]
            rms_s, gap_s, dmin_s = stds["rms"], stds["gap"], stds["dmin"]
            parts["latitude"] = (df_imp["latitude"] / lat_s) * self.GEO_WEIGHT
            parts["longitude"] = (df_imp["longitude"] / lon_s) * self.GEO_WEIGHT
            parts["rms"] = (df_imp["rms"] / rms_s) * self.OTHER_WEIGHT
            parts["gap"] = (df_imp["gap"] / gap_s) * self.OTHER_WEIGHT
            parts["dmin"] = (df_imp["dmin"] / dmin_s) * self.OTHER_WEIGHT
            return pd.DataFrame(parts, index=df_imp.index)

        for c in self.cols:
            s = stds[c]
            w = (
                self.GEO_WEIGHT
                if c in ("latitude", "longitude")
                else self.OTHER_WEIGHT
            )
            parts[c] = (df_imp[c] / s) * w
        return pd.DataFrame(parts, index=df_imp.index)

    def _unscale_frame(
        self, df_scaled: pd.DataFrame, stds: dict[str, float]
    ) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        if self._is_seismic_block():
            out["latitude"] = (
                df_scaled["latitude"] * stds["latitude"] / self.GEO_WEIGHT
            )
            out["longitude"] = (
                df_scaled["longitude"] * stds["longitude"] / self.GEO_WEIGHT
            )
            out["rms"] = df_scaled["rms"] * stds["rms"] / self.OTHER_WEIGHT
            out["gap"] = df_scaled["gap"] * stds["gap"] / self.OTHER_WEIGHT
            out["dmin"] = df_scaled["dmin"] * stds["dmin"] / self.OTHER_WEIGHT
            return pd.DataFrame(out, index=df_scaled.index)

        for c in self.cols:
            s = stds[c]
            w = (
                self.GEO_WEIGHT
                if c in ("latitude", "longitude")
                else self.OTHER_WEIGHT
            )
            out[c] = df_scaled[c] * s / w
        return pd.DataFrame(out, index=df_scaled.index)

    def _knn_fitputer(self, df: pd.DataFrame, weights: str = "distance") -> pd.DataFrame:
        df_imp = df[self.cols].copy()
        stds = self.ensure_std(df_imp)
        k_scaled = self._scale_frame(df_imp, stds)
        imputer = KNNImputer(n_neighbors=self.k, weights=weights)
        self._impt = imputer
        imputed_scaled = imputer.fit_transform(k_scaled)
        df_imputed_scaled = pd.DataFrame(
            imputed_scaled, columns=self.cols, index=k_scaled.index
        )
        return self._unscale_frame(df_imputed_scaled, stds)

    def _knn_transputer(self, df: pd.DataFrame, weights: str = "distance") -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Reference dataframe ``self.df`` is required for transform.")
        df_imp = df[self.cols].copy()
        self.std_cache = ImputationStats.calc_std(self.df[self.cols].copy(), self.cols)
        stds = self.std_cache
        k_scaled = self._scale_frame(df_imp, stds)
        if self._impt is None:
            raise ValueError("Imputer is not fitted; call ``_knn_fitputer`` first.")
        imputed_scaled = self._impt.transform(k_scaled)
        df_imputed_scaled = pd.DataFrame(
            imputed_scaled, columns=self.cols, index=k_scaled.index
        )
        return self._unscale_frame(df_imputed_scaled, stds)


class C_KNN:
    def __init__(
        self,
        dataframe,
        st_d=(),
        weights="distance",
        cols=None,
        num_neighbors=5,
    ) -> None:

        self.df = dataframe
        self.weights = weights
        self.std = st_d
        self.cols = list(DEFAULT_IMPUTE_COLS) if cols is None else list(cols)
        self.knn_data = KNNImpute(
            data=dataframe, num_neighbors=num_neighbors, cols=self.cols
        )
        self._impt = None
        self.std_cache = None

        self.ensure_std(self.df)

    def ensure_std(self, df_imp):
        """Compute std values once and reuse across methods."""
        if self.std_cache is None:
            self.std_cache = ImputationStats.calc_std(df_imp)
        return self.std_cache

    def _knn_fitputer(self, train_df, weights=None):
        w = self.weights if weights is None else weights
        out = self.knn_data._knn_fitputer(train_df, weights=w)
        self._impt = self.knn_data._impt
        self.std_cache = self.knn_data.std_cache
        return out

    def _knn_transputer(self, pred_df, weights=None):
        w = self.weights if weights is None else weights
        self.knn_data._impt = self._impt
        self.knn_data.std_cache = self.std_cache
        return self.knn_data._knn_transputer(pred_df, weights=w)

    def _knn_fit(self):
        return self._knn_fitputer(self.df)

    def _knn_transform(self, data):
        return self._knn_transputer(data)


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
        missing_pred_count=2699,
    ):
        self.x_t = x_t
        self.X_train_imp = X_train_imp
        self.X_valid_imp = X_valid_imp
        self.Data_Sets = Data_Sets
        self.ExperimentalDataPreprocessor = ExperimentalDataPreprocessor
        self.get_enc = get_enc
        self.ZScoreStandard = ZScoreStandard
        self.missing_pred_count = missing_pred_count

        self.fin_combined_df2_s_X_e = None
        self.train_df2_X = None
        self.train_df2_y = None
        self.pred_df2_X = None
        self.pred_df2_y = None
        self.train_df_filled2_X = None
        self.pred_df_filled2_X = None
        self.cols_r = None
        self.final_filled = None

        self._run_pipeline()

    def _concat_gap(self):
        gap_df = pd.concat(
            [self.X_train_imp[["gap"]], self.X_valid_imp[["gap"]]],
            axis=0,
        )
        return gap_df

    def _patch_missing_gap(self, gap_df):
        df_filled = self.x_t.copy()
        df_filled["gap"] = df_filled["gap"].fillna(gap_df["gap"])
        return df_filled

    def _split_by_dmin(self, df):
        train = df[df["dmin"].notnull()]
        pred = df[df["dmin"].isnull()]
        return train, pred

    def _recombine_and_sort(self, train, pred):
        return pd.concat([train, pred], axis=0).sort_index()

    def _process_features(self, fin_combined_df):
        _non_obj = fin_combined_df.select_dtypes(exclude=["object"])
        X, y = self.Data_Sets.split_dataset_xy(
            fin_combined_df, target_col="dmin", set_split=False
        )
        cols_r = X.select_dtypes(include=["float64"]).columns
        X_proc = self.ExperimentalDataPreprocessor(X).imput_data_transform(
            log_t=False, square=False, dmin_convert=False, cols=cols_r
        )
        X_enc = self.get_enc(X_proc)
        return X_enc, y, cols_r

    def _final_split_and_scale(self, X_enc, y, cols_r):
        split_idx = X_enc.shape[0] - self.missing_pred_count
        train_X = X_enc.iloc[:split_idx].copy()
        train_y = y.iloc[:split_idx].copy()
        pred_X = X_enc.iloc[split_idx:].copy()
        pred_y = y.iloc[split_idx:].copy()
        z_scorer = self.ZScoreStandard(train_X, cols_r)
        train_X_scaled = z_scorer.fit_standard_Z()
        pred_X_scaled = z_scorer.transform_standard_Z(pred_X)
        return train_X, train_y, pred_X, pred_y, train_X_scaled, pred_X_scaled

    def _run_pipeline(self):
        gap_df = self._concat_gap()
        self.final_filled = self._patch_missing_gap(gap_df)
        df_filled_train, df_filled_pred = self._split_by_dmin(self.final_filled)
        combined = self._recombine_and_sort(df_filled_train, df_filled_pred)
        combined_train, combined_pred = self._split_by_dmin(combined)
        fin_combined_df = pd.concat([combined_train, combined_pred], axis=0)

        X_enc, y, cols_r = self._process_features(fin_combined_df)
        self.fin_combined_df2_s_X_e = X_enc
        self.cols_r = cols_r

        (
            self.train_df2_X,
            self.train_df2_y,
            self.pred_df2_X,
            self.pred_df2_y,
            self.train_df_filled2_X,
            self.pred_df_filled2_X,
        ) = self._final_split_and_scale(X_enc, y, cols_r)


class RidgeRegressionImputer:
    def __init__(self, alpha=1.0):
        self.feature_cols = ["latitude", "longitude", "gap", "rms"]
        self.target_col = "dmin"
        self.model = Ridge(alpha=alpha)

    def fit(self, df_train):
        train_known = df_train[df_train[self.target_col].notna()]
        X_train = train_known[self.feature_cols].values
        y_train = train_known[self.target_col].values
        y_train_log = np.log(y_train + 1)
        self.model.fit(X_train, y_train_log)

    def impute(self, df_train):
        feature_cols = self.feature_cols
        target_col = self.target_col
        self.fit(df_train)
        train_known = df_train[df_train[target_col].notna()]
        df_test = df_train[df_train[target_col].isna()]
        train_missing_mask = df_train[target_col].isna()
        if train_missing_mask.any():
            X_miss = df_train.loc[train_missing_mask, feature_cols].values
            log_preds = self.model.predict(X_miss)
            df_train.loc[train_missing_mask, target_col] = np.exp(log_preds)
        test_missing_mask = df_test[target_col].isna()
        if test_missing_mask.any():
            X_miss = df_test.loc[test_missing_mask, feature_cols].values
            log_preds = self.model.predict(X_miss)
            df_test.loc[test_missing_mask, target_col] = np.exp(log_preds)
        return train_known, df_test


class ModelImputationOLD:
    def run_imputation_and_regression(self, data=None, save=False):
        print("Loading Data...")
        x_t = EQDataLoader().data_prep() if data is None else data
        print("Data Loaded Successfully")

        print("Initializing Data Split...")
        train_df, pred_df = Data_Sets.stratify_split(x_t)
        print("Data Splitting Successful")

        # Compute standard deviations using ImputationStats
        stds = ImputationStats.calc_std(
            train_df[DEFAULT_IMPUTE_COLS],
            DEFAULT_IMPUTE_COLS,
        )

        print("KNN for Imputation: Initialized\nImputing... Please wait")
        # knn_imp = C_KNN(
        #     dataframe=train_df,
        #     st_d=stds,
        #     cols=["latitude", "longitude", "dmin", "gap", "rms"],
        # )
        # knn_imp._knn_fit()
        # X_train_imp = knn_imp._knn_transform(train_df)
        # X_valid_imp = knn_imp._knn_transform(pred_df)

        knn_imp = KNNImpute(x_t)
        # knn_imp._knn_fit()
        X_train_imp = knn_imp._knn_fitputer(train_df)
        X_valid_imp = knn_imp._knn_transputer(pred_df)


        print("KNN for Imputation: Completed")

        print("Data Recalibration for Regression Imputation: Initialized")
        re_data = ModelDataReCourse(
            x_t=x_t,
            X_train_imp=X_train_imp,
            X_valid_imp=X_valid_imp,
            Data_Sets=Data_Sets, 
            ExperimentalDataPreprocessor=ExperimentalDataPreprocessor,
            get_enc=get_enc,
            ZScoreStandard=ZScoreStandard,
        )
        print("Data Recalibration for Regression Imputation: Successful")

        patched_train, patched_test = RidgeRegressionImputer().impute(re_data.final_filled)
        print("Regression Imputation: Successful")
        
        final_df = pd.concat([patched_train, patched_test], axis=0).sort_index()
        
        if save:
            # Save final_df to a CSV file in the data\engineered_data folder
            output_path = os.path.join(r'eq_prediction\data\engineered_data', "FinalData.csv")
            final_df.to_csv(output_path, index=False)
            print(f"final_df saved to {output_path}")
 
        # Return relevant objects for further use if desired
        return {
            "re_data": re_data,
            "p_train": patched_train,
            "p_test": patched_test,
            "Final_df": final_df,
            "kNN_imp_train": X_train_imp,
            "kNN_imp_valid": X_valid_imp,
        }


class ModelImputation:
    @staticmethod
    def _stratify_train_pred(
        df: pd.DataFrame, shuffle: bool = True
    ):
        if "gap" in df.columns:
            train_df, pred_df = Data_Sets.stratify_split(df, shuffle=shuffle)
            train_df = train_df.drop(columns=["gap_missing"], errors="ignore")
            pred_df = pred_df.drop(columns=["gap_missing"], errors="ignore")
            return train_df, pred_df
        return train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            shuffle=shuffle,
        )

    def impute_dataframe_numerical(
        self,
        data: pd.DataFrame,
        cols: list[str] | None = None,
        num_neighbors: int = 5,
        shuffle: bool = True,
    ) -> pd.DataFrame:
        num_cols = resolve_numeric_impute_columns(data, cols)
        if not num_cols:
            return data.copy()

        full = data.copy()
        train_df, pred_df = self._stratify_train_pred(full, shuffle=shuffle)

        knn = KNNImpute(data=full, num_neighbors=num_neighbors, cols=num_cols)
        x_train_imp = knn._knn_fitputer(train_df)
        x_pred_imp = knn._knn_transputer(pred_df)

        out = full.copy()
        for c in num_cols:
            out.loc[x_train_imp.index, c] = x_train_imp[c].to_numpy()
            out.loc[x_pred_imp.index, c] = x_pred_imp[c].to_numpy()
        return out

    def run_imputation_and_regression(
        self,
        data=None,
        save=False,
        num_neighbors: int = 5,
        ridge_alpha: float = 1.0,
        shuffle: bool = True,
    ):
        
        if data is None:
            raise ValueError("Notebook clone: pass ``data`` (e.g. ``fin_df``).")
        x_t = data.copy()
        train_df, pred_df = self._stratify_train_pred(x_t, shuffle=shuffle)

        knn_cols = resolve_numeric_impute_columns(x_t, cols=None)
        if not knn_cols:
            raise ValueError(
                "No KNN columns resolved; need numeric columns from DEFAULT_IMPUTE_COLS."
            )

        knn_imp = KNNImpute(data=x_t, num_neighbors=num_neighbors, cols=knn_cols)
        X_train_imp = knn_imp._knn_fitputer(train_df)
        X_valid_imp = knn_imp._knn_transputer(pred_df)

        final_filled = x_t.copy()
        if "gap" in knn_cols and "gap" in final_filled.columns:
            gap_df = pd.concat(
                [X_train_imp[["gap"]], X_valid_imp[["gap"]]], axis=0
            )
            final_filled["gap"] = final_filled["gap"].fillna(gap_df["gap"])

        need = {"latitude", "longitude", "gap", "rms", "dmin"}
        if not need <= set(final_filled.columns):
            raise ValueError(
                f"RidgeRegressionImputer needs all of {sorted(need)}; missing "
                f"{sorted(need - set(final_filled.columns))}."
            )

        patched_train, patched_test = RidgeRegressionImputer(
            alpha=ridge_alpha
        ).impute(final_filled)
        final_df = pd.concat([patched_train, patched_test], axis=0).sort_index()
        # print(final_df.head())

        if save:
            out_dir = Path("data/engineered_data")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "FinalData.csv"
            final_df.to_csv(out_path, index=False)
            print(f"final_df saved to {out_path.resolve()}")

        re_data = SimpleNamespace(final_filled=final_filled)
        return {
            "re_data": re_data,
            "p_train": patched_train,
            "p_test": patched_test,
            "Final_df": final_df,
            "kNN_imp_train": X_train_imp,
            "kNN_imp_valid": X_valid_imp,
        }


def run_test_funcs(X=None, save_Data=True):
    data = EQDataLoader(Saved=save_Data).extra_data_prep() if X is None else X

    cols_default = resolve_numeric_impute_columns(data, cols=None)
    print("Default KNN (seismic) columns:", cols_default)

    model_imp = ModelImputation()
    pipe = model_imp.run_imputation_and_regression(data=data, save=False)
    Xt = pipe["Final_df"]
    print(
        "KNN gap-patch + Ridge dmin — gap NA:",
        int(Xt["gap"].isna().sum()),
        "dmin NA:",
        int(Xt["dmin"].isna().sum()),
    )

    try:
        from IPython.display import display
    except ImportError:
        display = print

    display(Xt.head())
    num_cols = [
        c
        for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c])
        and not pd.api.types.is_bool_dtype(data[c])
    ]
    print("Missing counts after (numeric):", Xt[num_cols].isna().sum().to_dict())
    return Xt

if __name__ == "__main__":
    print(run_test_funcs())
    print(run_test_funcs().isna().sum())
