"""
train.py
---------
End-to-end training script:
  1. Load + clean raw data, engineer features
  2. Stratified train/test split (split BEFORE any preprocessing, to avoid leakage)
  3. Build a single sklearn Pipeline (preprocessing + model) per candidate model
  4. Train & compare Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
  5. Pick a "business-best" model (recall / ROC-AUC weighted, not raw accuracy)
  6. Hyperparameter-tune the winner with RandomizedSearchCV
  7. Refit final pipeline on the full training set, evaluate on held-out test set
  8. Save the COMPLETE pipeline (preprocessing + model, as one object) via joblib
  9. Save metrics, feature importances, and diagnostic plots to reports/

Run from the project root:  python src/train.py
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_raw_data, clean_data, data_quality_report
from feature_engineering import engineer_features
from evaluation import compute_metrics, get_classification_report, plot_confusion_matrix, plot_roc_curve, plot_pr_curve

RANDOM_STATE = 42
RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_PATH = "data/processed/churn_processed.csv"
MODEL_PATH = "models/churn_model.pkl"
REPORT_JSON_PATH = "reports/model_report.json"
REPORT_MD_PATH = "reports/model_report.md"
FIGURES_DIR = "reports/figures"

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "ServiceCount",
    "ContractRiskScore", "IsElectronicCheck", "SeniorCitizen",
    "HasTechSupport", "HasOnlineSecurity",
]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "TenureGroup",
    "MonthlyChargeCategory", "CustomerValueSegment",
]
TARGET = "ChurnFlag"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def get_candidate_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, max_depth=8
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---------- 1. Load, clean, engineer ----------
    raw = load_raw_data(RAW_PATH)
    dq_report = data_quality_report(raw)
    clean = clean_data(raw)
    full = engineer_features(clean)
    full.to_csv(PROCESSED_PATH, index=False)

    X = full[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = full[TARGET]

    # ---------- 2. Stratified split BEFORE any preprocessing ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # ---------- 3 & 4. Train & compare candidate models ----------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = get_candidate_models()
    comparison = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ])
        cv_roc_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["cv_roc_auc_mean"] = round(cv_roc_auc.mean(), 4)
        metrics["cv_roc_auc_std"] = round(cv_roc_auc.std(), 4)
        comparison[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"[{name}] {metrics}")

    # ---------- 5. Business-informed model selection ----------
    # Rationale: for churn, a False Negative (predicted "won't churn" but
    # customer actually leaves) means the retention team never reaches out
    # -> guaranteed lost revenue. A False Positive (predicted "will churn"
    # but customer actually stays) just costs a discount/outreach call to a
    # happy customer -> much cheaper mistake. So we weight RECALL and
    # ROC-AUC (ranking quality across all thresholds) above raw accuracy or
    # precision when choosing the model.
    def business_score(m):
        return 0.6 * m["roc_auc"] + 0.4 * m["recall"]

    best_model_name = max(comparison, key=lambda k: business_score(comparison[k]))
    print(f"\nSelected as business-best model: {best_model_name}")
    print(f"Reason: highest weighted score of ROC-AUC (ranking quality) and "
          f"Recall (catching actual churners), since missing a churner is "
          f"costlier than a false alarm.")

    # ---------- 6. Hyperparameter tuning of the winner ----------
    param_distributions = {
        "LogisticRegression": {
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "DecisionTree": {
            "classifier__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
            "classifier__min_samples_leaf": [1, 5, 10, 20, 30],
            "classifier__criterion": ["gini", "entropy"],
        },
        "RandomForest": {
            "classifier__n_estimators": [200, 300, 400, 500],
            "classifier__max_depth": [4, 6, 8, 10, 12, None],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "classifier__n_estimators": [100, 150, 200, 300],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "classifier__max_depth": [2, 3, 4, 5],
            "classifier__subsample": [0.7, 0.85, 1.0],
        },
    }

    base_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", get_candidate_models()[best_model_name]),
    ])

    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_distributions[best_model_name],
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    tuned_pipe = search.best_estimator_

    print(f"\nBest params for {best_model_name}: {search.best_params_}")
    print(f"Best CV ROC-AUC during tuning: {search.best_score_:.4f}")

    # ---------- 7. Final evaluation on held-out test set ----------
    y_pred_final = tuned_pipe.predict(X_test)
    y_proba_final = tuned_pipe.predict_proba(X_test)[:, 1]
    final_metrics = compute_metrics(y_test, y_pred_final, y_proba_final)
    final_report_text = get_classification_report(y_test, y_pred_final)
    print("\nFinal tuned model performance on held-out test set:")
    print(final_metrics)
    print(final_report_text)

    # ---------- Diagnostic plots ----------
    plot_confusion_matrix(y_test, y_pred_final, f"{FIGURES_DIR}/confusion_matrix.png",
                           title=f"{best_model_name} (tuned) — Confusion Matrix")
    plot_roc_curve(y_test, y_proba_final, f"{FIGURES_DIR}/roc_curve.png",
                   title=f"{best_model_name} (tuned) — ROC Curve")
    plot_pr_curve(y_test, y_proba_final, f"{FIGURES_DIR}/pr_curve.png",
                  title=f"{best_model_name} (tuned) — Precision-Recall Curve")

    # ---------- 8. Save the COMPLETE pipeline ----------
    joblib.dump(tuned_pipe, MODEL_PATH)
    print(f"\nSaved final pipeline (preprocessing + model) to {MODEL_PATH}")

    # ---------- Feature importance / coefficients ----------
    feature_names = tuned_pipe.named_steps["preprocessor"].get_feature_names_out()
    clf = tuned_pipe.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        importance_type = "feature_importances_ (impurity-based)"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        importance_type = "abs(coefficient) — Logistic Regression"
    else:
        importances = np.zeros(len(feature_names))
        importance_type = "unavailable"

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv("reports/feature_importance.csv", index=False)
    top15 = fi_df.head(15)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15["feature"][::-1], top15["importance"][::-1], color="#2b6cb0")
    ax.set_title(f"Top 15 Churn Drivers ({importance_type})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/feature_importance.png", dpi=120)
    plt.close(fig)

    # ---------- 9. Save full report (JSON + Markdown) ----------
    report = {
        "data_quality": dq_report,
        "model_comparison": comparison,
        "selected_model": best_model_name,
        "selection_rationale": (
            "Weighted score = 0.6*ROC-AUC + 0.4*Recall. A missed churner "
            "(false negative) costs the business a lost customer with no "
            "retention attempt, while a false positive only costs an "
            "unnecessary outreach/discount to a customer who would have "
            "stayed anyway — so recall and ranking quality (ROC-AUC) are "
            "weighted above raw accuracy."
        ),
        "best_hyperparameters": search.best_params_,
        "cv_roc_auc_during_tuning": round(search.best_score_, 4),
        "final_test_metrics": final_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_used": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)

    with open(REPORT_MD_PATH, "w") as f:
        f.write("# Model Report — Customer Churn Prediction\n\n")
        f.write("## Data Quality Summary\n\n")
        for k, v in dq_report.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Model Comparison (test set)\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC (mean±std) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, m in comparison.items():
            f.write(f"| {name} | {m['accuracy']} | {m['precision']} | {m['recall']} | "
                     f"{m['f1_score']} | {m['roc_auc']} | {m['cv_roc_auc_mean']} ± {m['cv_roc_auc_std']} |\n")
        f.write(f"\n## Selected Model: **{best_model_name}**\n\n")
        f.write(report["selection_rationale"] + "\n\n")
        f.write(f"## Hyperparameter Tuning\n\nBest params: `{search.best_params_}`\n\n")
        f.write(f"Best CV ROC-AUC during tuning: **{search.best_score_:.4f}**\n\n")
        f.write("## Final Test-Set Performance (tuned model)\n\n")
        for k, v in final_metrics.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n```\n" + final_report_text + "\n```\n")
        f.write("\n## Top Churn Drivers\n\n")
        try:
            f.write(top15.to_markdown(index=False))
        except Exception:
            f.write("| feature | importance |\n|---|---|\n")
            for _, row in top15.iterrows():
                f.write(f"| {row['feature']} | {row['importance']:.4f} |\n")
        f.write("\n\n*Note: this is model feature importance, not proof of causation. "
                "It shows which features the model relies on most to make predictions, "
                "which is correlational within the model, not a causal claim.*\n")

    print(f"\nSaved reports to {REPORT_JSON_PATH} and {REPORT_MD_PATH}")
    print(f"Selected model: {best_model_name} | Final test ROC-AUC: {final_metrics['roc_auc']}")


if __name__ == "__main__":
    main()
