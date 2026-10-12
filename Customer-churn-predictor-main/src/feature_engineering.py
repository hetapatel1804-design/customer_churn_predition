"""
feature_engineering.py
------------------------
Creates business-meaningful features from the cleaned dataframe.
None of these features use the target (`Churn`) in their computation,
so there is no target leakage.

Each feature below: WHAT it is -> WHY it helps -> BUSINESS MEANING.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

ADDON_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    TenureGroup: bucket raw tenure (months) into lifecycle stages.
    WHY: tenure's relationship with churn is non-linear (very new and
    very old customers behave differently); binning lets tree AND linear
    models pick up on that without over-fitting to exact month counts.
    BUSINESS MEANING: "New" customers are in the highest-risk onboarding
    window; "Loyal" customers are the most stable.
    """
    bins = [-1, 12, 24, 48, 60, 72]
    labels = ["0-12 (New)", "13-24 (Early)", "25-48 (Established)",
              "49-60 (Loyal)", "61-72 (Veteran)"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


def add_monthly_charge_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    MonthlyChargeCategory: Low / Medium / High, based on the dataset's
    own tercile (33rd/66th percentile) cut points — computed from data,
    not arbitrary guesses.
    WHY: raw dollar amounts are harder for stakeholders to reason about
    than a simple tier; also helps linear models capture non-linear
    price sensitivity.
    BUSINESS MEANING: flags customers on premium plans who may be
    price-sensitive / re-shopping competitors.
    """
    q1, q2 = df["MonthlyCharges"].quantile([1 / 3, 2 / 3])
    def bucket(x):
        if x <= q1:
            return "Low"
        elif x <= q2:
            return "Medium"
        return "High"
    df["MonthlyChargeCategory"] = df["MonthlyCharges"].apply(bucket)
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    ServiceCount: number of add-on services (security, backup, device
    protection, tech support, streaming TV, streaming movies) the
    customer has subscribed to (0-6).
    WHY: bundled/engaged customers are known in subscription businesses
    to churn less — this captures "stickiness" in one number.
    BUSINESS MEANING: low ServiceCount = shallow relationship with the
    company = easier to leave.
    """
    df["ServiceCount"] = (df[ADDON_SERVICE_COLS] == "Yes").sum(axis=1)
    return df


def add_has_support_and_security(df: pd.DataFrame) -> pd.DataFrame:
    """
    HasTechSupport / HasOnlineSecurity: binary flags.
    WHY: these two specific services showed the strongest raw
    association with retention in EDA — isolating them as explicit
    binary flags makes their effect easier for both models and humans
    to read, on top of the aggregate ServiceCount.
    BUSINESS MEANING: customers without support/security services hit
    friction alone and are more likely to leave when something goes wrong.
    """
    df["HasTechSupport"] = (df["TechSupport"] == "Yes").astype(int)
    df["HasOnlineSecurity"] = (df["OnlineSecurity"] == "Yes").astype(int)
    return df


def add_contract_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    ContractRiskScore: ordinal risk implied by contract length
    (Month-to-month=3 > One year=2 > Two year=1).
    WHY: encodes the natural ordering of commitment level, which a
    plain one-hot encoding discards; gives linear models a monotonic
    signal to work with.
    BUSINESS MEANING: shorter commitment = lower switching cost = higher
    churn risk.
    """
    mapping = {"Month-to-month": 3, "One year": 2, "Two year": 1}
    df["ContractRiskScore"] = df["Contract"].map(mapping)
    return df


def add_payment_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    IsElectronicCheck: binary flag for the "Electronic check" payment
    method specifically (vs. automatic bank/credit methods or mailed
    check).
    WHY: electronic check is a manual, non-automatic payment method and
    is well known in this dataset to co-occur heavily with churn — a
    single flag isolates that signal cleanly.
    BUSINESS MEANING: customers not on auto-pay have to actively
    re-decide to keep paying every cycle, raising the chance they
    reconsider and leave.
    """
    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    return df


def add_customer_value_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    CustomerValueSegment: Low/Medium/High based on tercile cut points of
    TotalCharges (cumulative revenue to date).
    WHY: lets the business (and the model) distinguish "small, new,
    cheap accounts" from "long-standing, high-value accounts" — losing
    the latter is more costly.
    BUSINESS MEANING: powers the "prioritize retention effort by value"
    logic in the dashboard (Revenue at Risk).
    """
    q1, q2 = df["TotalCharges"].quantile([1 / 3, 2 / 3])
    def bucket(x):
        if x <= q1:
            return "Low"
        elif x <= q2:
            return "Medium"
        return "High"
    df["CustomerValueSegment"] = df["TotalCharges"].apply(bucket)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df = df.copy()
    df = add_tenure_group(df)
    df = add_monthly_charge_category(df)
    df = add_service_count(df)
    df = add_has_support_and_security(df)
    df = add_contract_risk_score(df)
    df = add_payment_risk_flag(df)
    df = add_customer_value_segment(df)
    return df


if __name__ == "__main__":
    from data_preprocessing import load_raw_data, clean_data
    raw = load_raw_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    clean = clean_data(raw)
    fe = engineer_features(clean)
    print(fe[["TenureGroup", "MonthlyChargeCategory", "ServiceCount",
               "ContractRiskScore", "IsElectronicCheck",
               "CustomerValueSegment"]].head())
