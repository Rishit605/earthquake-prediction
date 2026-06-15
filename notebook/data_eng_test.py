import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.helpers.datapi import callDataFetcher
from src.preprocessing.feature_eng import FilterAndFill

DATA_PATH = PROJECT_ROOT / "data" / "engineered_data" / "New_Engineered_Data.csv"


def _resolve_output_path(file_name=None):
    if file_name is None:
        return DATA_PATH

    output_path = Path(file_name)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".csv")
    if not output_path.is_absolute():
        output_path = DATA_PATH.parent / output_path
    return output_path


def _resolve_existing_path(existing_file_name=None):
    if existing_file_name is None:
        return DATA_PATH

    existing_path = Path(existing_file_name)
    if existing_path.suffix == "":
        existing_path = existing_path.with_suffix(".csv")
    if not existing_path.is_absolute():
        existing_path = DATA_PATH.parent / existing_path
    return existing_path


def data_eng_test(data=None, max_workers=3, file_name=None):
    from src.preprocessing.data_preprocessing import drop_rate_new
    if data is not None:
        df = data
    else:
        df = callDataFetcher(True)
    cols_to_drop = drop_rate_new(df)

    missing_data_df = (
        df.loc[df["nst"].isna()]
        .drop(columns=cols_to_drop, errors="ignore")
        .reset_index(names="old_idx")
    )

    print(missing_data_df.shape[0], "original rows to process before excluding existing data.")
    return 
    
    
    if not missing_data_df.empty:
        missing_data_df = FilterAndFill().filter_and_fill(
            missing_data_df, max_workers=max_workers
        )

    output_path = _resolve_output_path(file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_data_df.to_csv(output_path, index=False)
    return missing_data_df


def data_eng_test_incremental(
    data=None,
    max_workers=3,
    file_name="New_Engineered_Data_Patch.csv",
    existing_file_name=None,
):
    from src.preprocessing.data_preprocessing import drop_rate_new

    if data is not None:
        df = data
    else:
        df = callDataFetcher(True)

    cols_to_drop = drop_rate_new(df)
    missing_data_df = (
        df.loc[df["nst"].isna()]
        .drop(columns=cols_to_drop, errors="ignore")
        .reset_index(names="old_idx")
    )

    existing_path = _resolve_existing_path(existing_file_name)
    if existing_path.exists():
        import pandas as pd

        existing_data_df = pd.read_csv(existing_path)
        if "old_idx" not in existing_data_df.columns:
            raise ValueError(
                f"Expected existing engineered data at {existing_path} to contain "
                "an 'old_idx' column."
            )
        existing_indices = set(existing_data_df["old_idx"].dropna())
    else:
        existing_indices = set()

    missing_data_df = missing_data_df[
        ~missing_data_df["old_idx"].isin(existing_indices)
    ].reset_index(drop=True)
    
    print(missing_data_df.shape[0], "new rows to process after excluding existing data.")
    return 

    if not missing_data_df.empty:
        missing_data_df = FilterAndFill().filter_and_fill(
            missing_data_df, max_workers=max_workers
        )

    output_path = _resolve_output_path(file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_data_df.to_csv(output_path, index=False)
    return missing_data_df


if __name__ == "__main__":
    from src.helpers.datapi import callDataFetcher
    data = callDataFetcher(True)
    print(data.shape)
    print(data.isna().sum())
    # data_eng_test(data=data, file_name="New_Engineered_Data2.csv")
    # data_eng_test_incremental(data=data, existing_file_name="New_Engineered_Data3.csv")