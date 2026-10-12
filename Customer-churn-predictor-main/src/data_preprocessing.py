"""
data_preprocessing.py
----------------------
Loads and cleans the raw Telco Customer Churn dataset.

Design decisions (documented, not silent):
1. `TotalCharges` is stored as a string in the raw file. 11 rows contain a
   blank string (" ") instead of a number. Investigation shows every one of
   these rows has `tenure == 0` (brand new customers who have not been
   billed yet). We therefore set TotalCharges = 0.0 for these rows rather
   than mean/median-imputing, because 0 is the structurally correct value,
   not a "missing" one.
2. `customerID` is kept in the dataframe (useful for reporting / dashboard
   lookups) but is explicitly excluded from the model's feature set in
   train.py — it has no predictive value and including it would be a
   leakage / overfitting risk (unique per row).
3. No rows are dropped. There are 0 duplicate customerIDs and 0 fully
   duplicated rows in the raw file (verified), so no de-duplication is
   required.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

RAW_TARGET_COL = "Churn"


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as shipped by the data source."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented cleaning steps and return a clean copy."""
    df = df.copy()

    # --- Fix TotalCharges dtype + blank-string issue -----------------
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    blank_mask = df["TotalCharges"] == ""
    df.loc[blank_mask, "TotalCharges"] = "0"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="raise")

    # Sanity check: every row we zero-filled should have tenure == 0.
    # If that assumption ever breaks (e.g. dataset updated), we want to
    # know rather than silently mis-impute.
    zero_rows = df.loc[blank_mask]
    if not (zero_rows["tenure"] == 0).all():
        raise ValueError(
            "Found blank TotalCharges rows with tenure != 0 — "
            "zero-fill assumption no longer holds, review manually."
        )

    # --- Normalize SeniorCitizen to Yes/No for readability in EDA -----
    # (Model training uses the numeric 0/1 version — see feature_engineering.py)
    df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # --- Strip stray whitespace from all string/object columns --------
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # --- Basic dtype enforcement ---------------------------------------
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # --- Target to binary int (kept as a separate column, original text
    #     label is preserved for readability in reports/EDA) -------------
    df["ChurnLabel"] = df[RAW_TARGET_COL]
    df["ChurnFlag"] = (df[RAW_TARGET_COL] == "Yes").astype(int)

    return df


def data_quality_report(df_raw: pd.DataFrame) -> dict:
    """Compute real data-quality stats from the raw dataframe (no invented numbers)."""
    total_charges_blank = (df_raw["TotalCharges"].astype(str).str.strip() == "").sum()
    report = {
        "n_rows": int(len(df_raw)),
        "n_cols": int(df_raw.shape[1]),
        "duplicate_customer_ids": int(df_raw["customerID"].duplicated().sum()),
        "duplicate_rows": int(df_raw.duplicated().sum()),
        "missing_values_per_column": df_raw.isnull().sum().to_dict(),
        "total_charges_blank_strings": int(total_charges_blank),
        "churn_value_counts": df_raw["Churn"].value_counts().to_dict(),
        "churn_rate_pct": round(
            (df_raw["Churn"] == "Yes").mean() * 100, 2
        ),
    }
    return report


if __name__ == "__main__":
    raw = load_raw_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(data_quality_report(raw))
    cleaned = clean_data(raw)
    print(cleaned.dtypes)
