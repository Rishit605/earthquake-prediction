from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Plot training and evaluation history
def plot_loss(train_losses, val_losses=None):
    if not val_losses:
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Evaluation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss History')
        plt.legend()
        plt.show()

# Print JSON Data of unknown structure
def walk_json(data, indent=0):
    if isinstance(data, dict):
        for key, value in data.items():
            print("  " * indent + str(key) + ":")
            walk_json(value, indent + 1)
    elif isinstance(data, list):
        for item in data:
            walk_json(item, indent + 1)
    else:
        print("  " * indent + str(data))

# Usage
# walk_json(geojson_data)


class DataDist:
    @staticmethod
    def get_kurtosis_for_all(data):
        from scipy.stats import kurtosis
        cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        rep = {}
        for col in cols:
            rep[col] = kurtosis(data[col])
        return rep

    @staticmethod
    def get_skew_for_all(data):
        from scipy.stats import skew
        cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        rep = {}
        for col in cols:
            rep[col] = skew(data[col])
        return rep


def plot_histograms(dataframe, columns=None):
    """
    Plots log-histograms with KDE for specified columns of the dataframe.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing numerical columns.
        columns (list or None): List of columns to plot. If None, defaults to ['mag', 'rms', 'dmin', 'elevation', 'gap'].
    """
    if columns is None:
        columns = ['mag', 'rms', 'dmin', 'elevation', 'gap']
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    import math
    n_cols = len(columns)
    n_rows = math.ceil(math.sqrt(n_cols))
    n_cols_per_row = math.ceil(n_cols / n_rows)
    print(n_cols, n_rows, n_cols_per_row)
    fig, axs = plt.subplots(n_rows, n_cols_per_row, figsize=(4 * n_cols_per_row, 4 * n_rows))
    axs = np.array(axs).flatten()

    for i, col in enumerate(columns):
        if col not in dataframe.columns:
            axs[i].set_visible(False)
            continue
        vals = dataframe[col].dropna()  
        positive = vals[vals > 0]
        if len(positive) == 0:
            axs[i].set_visible(False)
            continue
        sns.histplot(np.log(positive), kde=True, ax=axs[i])
        axs[i].set_title(f'Histogram of {col}')

    plt.tight_layout()
    plt.show()


## FINDING MOMENTS OR uNIVARIATE DISTRIBUTION
import numpy as np

def find_mean(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return np.nan
    return a.sum() / a.size

def find_variance(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return np.nan
    mean = find_mean(a)
    s = np.sum(np.square(a - mean))
    return s / a.size


def check_non_numeric_cols(df):
    """
    Checks for columns in the DataFrame with dtype other than float or int.
    Prints the results for non-numeric columns.
    """
    non_numeric_cols = [
        col for col in df.select_dtypes(exclude="object").columns
        if not pd.api.types.is_float_dtype(df[col]) and not pd.api.types.is_integer_dtype(df[col])
    ]
    if non_numeric_cols:
        print("Non-numeric columns detected:")
        for col in non_numeric_cols:
            print(f"Column '{col}' has dtype {df[col].dtype}")
    else:
        print("All columns are float or int.")

# Example usage:
# check_non_numeric_cols(data_loader)


def r2_Loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Residual sum of squares (SSE)
    ss_res = np.sum((y_true - y_pred) ** 2)
    # Total sum of squares (SST)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)