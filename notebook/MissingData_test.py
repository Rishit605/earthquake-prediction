import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.helpers.datapi import callDataFetcher
from src.preprocessing.feature_eng import FilterAndFill

ENGINEERED_DATA_PATH = (
    PROJECT_ROOT / "data" / "engineered_data" / "New_Engineered_Data.csv"
)
FOCUS_COLS = ["gap", "dmin"]


def patch_engineered_data(raw_df, engineered_path=ENGINEERED_DATA_PATH):
    engineered_df = pd.read_csv(engineered_path)
    if "old_idx" not in engineered_df.columns:
        raise ValueError(
            f"Expected engineered data at {engineered_path} to contain an 'old_idx' column."
        )

    patched_df = raw_df.copy()
    engineered_df = engineered_df.set_index("old_idx")

    overlap_cols = [
        col for col in engineered_df.columns if col in patched_df.columns
    ]
    if not overlap_cols:
        return patched_df

    patch_df = engineered_df[overlap_cols]
    patched_df[overlap_cols] = patched_df[overlap_cols].combine_first(patch_df)
    return patched_df


def summarize_focus_missing(raw_df, patched_df, focus_cols=FOCUS_COLS):
    rows = []
    for col in focus_cols:
        if col not in raw_df.columns or col not in patched_df.columns:
            rows.append(
                {
                    "column": col,
                    "raw_missing": None,
                    "patched_missing": None,
                    "reduction": None,
                }
            )
            continue

        raw_missing = int(raw_df[col].isna().sum())
        patched_missing = int(patched_df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "raw_missing": raw_missing,
                "patched_missing": patched_missing,
                "reduction": raw_missing - patched_missing,
            }
        )
    return pd.DataFrame(rows).set_index("column")


# TODO: Now implement the FilterAndFill logic and run the full incremental patching process, then compare results again.



def main():
    raw_df = callDataFetcher(True)
    patched_df = patch_engineered_data(raw_df)
    # Apply FilterAndFill to the patched dataframe. Support several possible
    # APIs for FilterAndFill (callable, transformer with fit_transform/transform).
    def _apply_filter_and_fill(df):
        missing_data_df = FilterAndFill().filter_and_fill(patched_df, max_workers=3)
        output_path = PROJECT_ROOT / "data" / "testfill_df.csv"
        missing_data_df.to_csv(output_path, index=False)
        return missing_data_df

    filtered_df = _apply_filter_and_fill(patched_df)

    print("Raw data missing values:")
    print(raw_df.isna().sum())
    print()

    print("Patched data missing values:")
    print(patched_df.isna().sum())
    print()

    print("Filtered (FilterAndFill) data missing values:")
    print(filtered_df.isna().sum())
    print()

    print("Focused missing-value comparison (raw -> patched -> filtered):")
    print("Raw vs Patched:")
    print(summarize_focus_missing(raw_df, patched_df))
    print()
    print("Patched vs Filtered:")
    print(summarize_focus_missing(patched_df, filtered_df))


if __name__ == "__main__":
    # main()
    # raw_df = callDataFetcher(True)
    # patched_df = patch_engineered_data(raw_df)
    # print(patched_df.head())
    
    ll = pd.read_csv(ENGINEERED_DATA_PATH)
    ll2 = pd.read_csv(PROJECT_ROOT / "data" / "engineered_data" / "FinalRegressionData.csv")
    ll3 = callDataFetcher(True)
    # print(ll.head())
    # print(ll.shape)
    # print(ll.isna().sum())
    # print()
    # print(ll2.head())
    # print(ll2.shape)
    # print(ll2.isna().sum())

    print(ll2.head())
    print(ll2.shape)
    print(ll2.isna().sum())
