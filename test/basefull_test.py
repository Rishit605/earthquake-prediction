from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "eq_data_updated2.csv"
REGRESSION_DATA_PATH = (
    PROJECT_ROOT / "data" / "engineered_data" / "FinalRegressionData.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "eq_data_updated3_patched.csv"

PATCH_COLUMNS = ["dmin", "gap", "rms", "nst"]


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    raw_df = df.copy()
    raw_df.reset_index(inplace=True)
    raw_df = raw_df.rename(columns={"index": "idx"})
    
    regression_df = pd.read_csv(REGRESSION_DATA_PATH)

    if "idx" not in regression_df.columns:
        raise KeyError("FinalRegressionData.csv must contain an 'idx' column.")
    if not regression_df["idx"].is_unique:
        raise ValueError("FinalRegressionData.csv contains duplicate 'idx' values.")

    regression_df = regression_df.set_index("idx")
    available_columns = [
        column
        for column in PATCH_COLUMNS
        if column in raw_df.columns and column in regression_df.columns
    ]
    unavailable_columns = [
        column for column in PATCH_COLUMNS if column not in regression_df.columns
    ]

    before_patch = raw_df[PATCH_COLUMNS].copy()

    # Series.fillna aligns the regression values to the raw row index.
    for column in available_columns:
        raw_df[column] = raw_df[column].fillna(regression_df[column])

    # Existing values must remain unchanged.
    for column in available_columns:
        existing_values = before_patch[column].notna()
        assert raw_df.loc[existing_values, column].equals(
            before_patch.loc[existing_values, column]
        )

    comparison = pd.DataFrame(
        {
            "before": before_patch.isna().sum(),
            "after": raw_df[PATCH_COLUMNS].isna().sum(),
            "filled": (
                before_patch.isna().sum()
                - raw_df[PATCH_COLUMNS].isna().sum()
            ),
        }
    )

    print("Raw data shape:", raw_df.shape)
    print("Patched columns:", available_columns)
    print("\nMissing-value comparison:")
    print(comparison)

    if unavailable_columns:
        print(
            "\nCould not patch from FinalRegressionData.csv because these "
            f"columns are absent: {unavailable_columns}"
        )

    raw_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved patched raw data to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    

