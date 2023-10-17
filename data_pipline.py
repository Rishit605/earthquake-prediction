from typing import Any, Dict, List
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Load the data
def load_data(data_path: str, unwanted_cols: List) -> pd.DataFrame:
    """Loads the data from the specified file path and returns a DataFrame.

    Args:
        data_path: The path to the data file.

    Returns:
        A DataFrame containing the data.

    Raises:
        ValueError: If the file format is not supported.
    """

    spl = data_path.split('.')
    if spl[1] == "xlsx":
        return pd.read_excel(data_path, header=1).drop(unwanted_cols, axis=1)
    elif spl[1] == "csv":
        return pd.read_csv(data_path).drop(unwanted_cols, axis=1)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


# Scale the data
def scaled(init_dat: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaler.fit(init_dat)

    rescl_data = pd.DataFrame(scaler.transform(init_dat),
                              columns=init_dat.columns,
                              index=init_dat.index)

    return rescl_data


# Splitting the dataset - For Re-Training
def split_data(inputs: pd.DataFrame, outputs: pd.Series, test_ratio: float) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs,
                                                        test_size=test_ratio, random_state=42)

    return {'X_TRAIN': X_train, 'Y_TRAIN': y_train,
            'X_TEST': X_test, 'Y_TEST': y_test}
