import os
import sys

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.helpers.datapi import url_data_call, callDataFetcher
from src.preprocessing.data_preprocessing import drop_rate
from src.preprocessing.feature_eng import FilterAndFill

DATA_PATH = PROJECT_ROOT / "data" / "engineered_data" / "New_Engineered_Data.csv"


def data_eng_test():
    df = callDataFetcher(True)
    cols_to_drop = drop_rate(df)
    dropped_df = df.drop(columns=cols_to_drop)

    missing_data_df = dropped_df[dropped_df['nst'].isna()]
    missing_data_df.reset_index(inplace=True)
    missing_data_df.rename(columns={'index': 'old_idx'}, inplace=True)
    # print(missing_data_df.head())
    
    missing_data_df = FilterAndFill().filter_and_fill_debug_new_change(missing_data_df, max_workers=2)

    missing_data_df.to_csv(DATA_PATH, index=True)
    return missing_data_df


if __name__ == "__main__":
    data_eng_test()