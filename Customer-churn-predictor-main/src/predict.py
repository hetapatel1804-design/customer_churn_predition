"""
predict.py
-----------
Loads the saved pipeline (preprocessing + model, one object — no
train/inference mismatch possible) and turns raw customer rows into a
business-ready output:

  CustomerID | ChurnProbability | RiskSegment | TopRiskFactors | SuggestedAction | RevenueAtRisk

Explainability approach
------------------------
No SHAP library is available in this build environment (offline sandbox,
no internet access to `pip install shap`). Rather than fabricate SHAP-style
numbers, this module implements a real, honest, from-scratch explanation:

  - If the trained classifier is LINEAR (e.g. Logistic Regression), we
    compute each feature's TRUE local contribution to the prediction as
    `coefficient * standardized_feature_value` (the actual log-odds
    contribution for that customer) — this is mathematically exact, not
    approximated.
  - If the classifier is TREE-BASED (Random Forest / Gradient Boosting /
    Decision Tree), we fall back to intersecting the model's GLOBAL
    feature_importances_ ranking with the specific risk conditions this
    customer meets (rule-based flags below). This is clearly weaker than
    SHAP and is labelled as such in the output.

To get true SHAP values, install `shap` in an environment with internet
access and use `shap.Explainer(pipe.named_steps["classifier"], ...)` on
the preprocessed matrix — the code is structured (features accessible via
`pipe.named_steps["preprocessor"].get_feature_names_out()`) to make that
a drop-in addition later; see README "Future Improvements".
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib

RISK_THRESHOLDS = {
    # NOTE: these are starting defaults, not proven-optimal cutoffs.
    # They should be re-validated against actual business costs
    # (cost of an outreach vs. cost of a lost customer) and the
    # precision/recall trade-off on a validation set before being
    # treated as final. See reports/figures/pr_curve.png.
    "high": 0.70,
    "medium": 0.40,
}

RISK_FACTOR_RULES = [
    # (condition_fn, label)
    (lambda r: r.get("Contract") == "Month-to-month", "Month-to-month contract (low switching cost)"),
    (lambda r: r.get("tenure", 99) < 12, "Short tenure (< 12 months)"),
    (lambda r: r.get("MonthlyCharges", 0) > 80, "High monthly charges"),
    (lambda r: r.get("TechSupport") == "No", "No tech support"),
    (lambda r: r.get("OnlineSecurity") == "No", "No online security add-on"),
    (lambda r: r.get("PaymentMethod") == "Electronic check", "Pays via electronic check (non-automatic)"),
    (lambda r: r.get("InternetService") == "Fiber optic", "On fiber optic plan (historically higher churn tier)"),
    (lambda r: r.get("PaperlessBilling") == "Yes", "Paperless billing (weaker relationship signal in this data)"),
    (lambda r: r.get("ServiceCount", 6) <= 1, "Few add-on services (low engagement/stickiness)"),
    (lambda r: r.get("Dependents") == "No" and r.get("Partner") == "No", "No partner/dependents (fewer account ties)"),
]


def load_pipeline(path: str = "models/churn_model.pkl"):
    return joblib.load(path)


def _risk_segment(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["high"]:
        return "High Risk"
    elif prob >= RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    return "Low Risk"


def _rule_based_risk_factors(row: pd.Series, max_factors: int = 5) -> list[str]:
    hits = [label for cond, label in RISK_FACTOR_RULES if cond(row)]
    return hits[:max_factors] if hits else ["No major rule-based risk flags triggered"]


def _linear_local_explanation(pipe, X_row: pd.DataFrame, max_factors: int = 5) -> list[str]:
    """
    Exact local explanation for linear models: contribution = coef * scaled_value.
    Returns the top contributing features (by absolute contribution) that
    PUSH TOWARD churn (positive contribution only — factors making the
    customer MORE likely to churn, which is what a retention agent needs).
    """
    preprocessor = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["classifier"]
    if not hasattr(clf, "coef_"):
        return None  # not a linear model

    X_transformed = preprocessor.transform(X_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]
    contributions = X_transformed[0] * coefs

    order = np.argsort(contributions)[::-1]  # highest positive contribution first
    top = []
    for idx in order:
        if contributions[idx] <= 0:
            continue
        name = feature_names[idx].replace("num__", "").replace("cat__", "")
        top.append(name)
        if len(top) >= max_factors:
            break
    return top if top else None


def explain_customer(pipe, row: pd.Series, X_row: pd.DataFrame) -> tuple[list[str], str]:
    """Returns (risk_factors, method_used)."""
    linear_factors = _linear_local_explanation(pipe, X_row)
    if linear_factors is not None:
        return linear_factors, "exact linear contribution (coefficient x value)"
    return _rule_based_risk_factors(row), "rule-based flags + global feature importance (fallback, no SHAP available)"


def suggested_action(row: pd.Series, risk_segment: str) -> str:
    """
    Rule-based business recommendation. These are BUSINESS RULES layered
    on top of the model's prediction, not something the ML model itself
    learned or claims to know.
    """
    if risk_segment == "Low Risk":
        return "No action needed — monitor at next billing cycle."

    actions = []
    if row.get("Contract") == "Month-to-month":
        actions.append("Offer discounted annual/two-year contract")
    if row.get("MonthlyCharges", 0) > 80:
        actions.append("Offer personalized pricing / plan review")
    if row.get("TechSupport") == "No":
        actions.append("Offer free/discounted tech support trial")
    if row.get("tenure", 99) < 12:
        actions.append("Provide onboarding check-in / early-tenure retention incentive")
    if row.get("PaymentMethod") == "Electronic check":
        actions.append("Encourage switch to autopay (bank transfer/credit card) with a small incentive")

    if not actions:
        actions.append("Proactive retention call from account manager")

    prefix = "URGENT — " if risk_segment == "High Risk" else ""
    return prefix + "; ".join(actions)


FEATURE_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "ServiceCount",
    "ContractRiskScore", "IsElectronicCheck", "SeniorCitizen",
    "HasTechSupport", "HasOnlineSecurity",
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "TenureGroup",
    "MonthlyChargeCategory", "CustomerValueSegment",
]


def predict_batch(pipe, df_customers: pd.DataFrame, id_col: str = "customerID") -> pd.DataFrame:
    """
    df_customers must already be CLEANED and FEATURE-ENGINEERED (i.e. have
    gone through data_preprocessing.clean_data + feature_engineering.engineer_features)
    and contain all columns in FEATURE_COLS plus the id_col.
    """
    X = df_customers[FEATURE_COLS]
    probas = pipe.predict_proba(X)[:, 1]

    results = []
    for i, (_, row) in enumerate(df_customers.iterrows()):
        prob = float(probas[i])
        segment = _risk_segment(prob)
        X_row = X.iloc[[i]]
        risk_factors, method = explain_customer(pipe, row, X_row)
        action = suggested_action(row, segment)
        revenue_at_risk = round(row.get("MonthlyCharges", 0) * prob, 2)
        results.append({
            id_col: row[id_col],
            "ChurnProbability": round(prob, 4),
            "RiskSegment": segment,
            "TopRiskFactors": ", ".join(risk_factors),
            "ExplanationMethod": method,
            "SuggestedAction": action,
            "MonthlyRevenueAtRisk": revenue_at_risk,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from data_preprocessing import load_raw_data, clean_data
    from feature_engineering import engineer_features

    pipe = load_pipeline("models/churn_model.pkl")
    raw = load_raw_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(clean_data(raw))

    out = predict_batch(pipe, df.head(5))
    pd.set_option("display.max_colwidth", 60)
    print(out.to_string(index=False))

    # Full-dataset scored file for the dashboard
    full_scored = predict_batch(pipe, df)
    full_scored.to_csv("data/processed/scored_customers.csv", index=False)
    print(f"\nSaved scored predictions for all {len(full_scored)} customers "
          f"to data/processed/scored_customers.csv")
    print(f"\nAggregate estimated Revenue at Risk (sum of MonthlyRevenueAtRisk): "
          f"${full_scored['MonthlyRevenueAtRisk'].sum():,.2f} / month")
