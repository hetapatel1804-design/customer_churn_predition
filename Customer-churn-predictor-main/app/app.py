"""
app.py — Customer Churn Prediction Dashboard (Streamlit)

Run with:  streamlit run app/app.py   (from the project root)

Pages:
  1. Executive Overview   — headline KPIs for leadership
  2. Customer Risk        — look up an individual customer
  3. Churn Analytics       — segment-level churn patterns
  4. Prediction            — score a brand-new (hypothetical) customer
  5. Model Evaluation      — ML performance metrics, for the data team
"""

import os
import sys
import json

import joblib
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_preprocessing import load_raw_data, clean_data
from feature_engineering import engineer_features
from predict import load_pipeline, predict_batch, FEATURE_COLS, RISK_THRESHOLDS, suggested_action, _risk_segment

# ----------------------------------------------------------------------
# Page config & light styling
# ----------------------------------------------------------------------
st.set_page_config(page_title="Churn Prediction System", page_icon="📉", layout="wide")

PRIMARY = "#2b6cb0"
DANGER = "#c53030"
WARN = "#dd6b20"
OK = "#2f855a"

st.markdown(
    f"""
    <style>
    .metric-card {{ padding: 1rem; border-radius: 8px; background: #f7fafc; border: 1px solid #e2e8f0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Cached data / model loading
# ----------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return load_pipeline("models/churn_model.pkl")


@st.cache_data
def get_scored_customers():
    path = "data/processed/scored_customers.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fall back to scoring on the fly if the pre-scored file isn't there
    raw = load_raw_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(clean_data(raw))
    pipe = get_pipeline()
    return predict_batch(pipe, df)


@st.cache_data
def get_processed_customers():
    return pd.read_csv("data/processed/churn_processed.csv")


@st.cache_data
def get_model_report():
    with open("reports/model_report.json") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("📉 Churn Prediction System")
page = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "Customer Risk", "Churn Analytics", "Prediction", "Model Evaluation"],
)

scored = get_scored_customers()
customers = get_processed_customers()
merged = customers.merge(scored, on="customerID", how="left")

# ========================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ========================================================================
if page == "Executive Overview":
    st.title("Executive Overview")
    st.caption("Headline churn metrics for leadership — updated from the latest scored customer base.")

    total_customers = len(merged)
    churn_rate = (customers["ChurnFlag"].mean()) * 100
    high_risk_count = (scored["RiskSegment"] == "High Risk").sum()
    avg_prob = scored["ChurnProbability"].mean()
    revenue_at_risk = scored["MonthlyRevenueAtRisk"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Historical Churn Rate", f"{churn_rate:.1f}%")
    c3.metric("Predicted High-Risk Customers", f"{high_risk_count:,}")
    c4.metric("Avg. Churn Probability", f"{avg_prob:.1%}")
    c5.metric("Est. Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Segment Breakdown")
        seg_counts = scored["RiskSegment"].value_counts()
        fig, ax = plt.subplots()
        colors = {"High Risk": DANGER, "Medium Risk": WARN, "Low Risk": OK}
        ax.bar(seg_counts.index, seg_counts.values,
               color=[colors.get(s, PRIMARY) for s in seg_counts.index])
        ax.set_ylabel("Number of customers")
        st.pyplot(fig)

    with col2:
        st.subheader("Revenue at Risk by Segment")
        rev_by_seg = scored.groupby("RiskSegment")["MonthlyRevenueAtRisk"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        ax.bar(rev_by_seg.index, rev_by_seg.values,
               color=[colors.get(s, PRIMARY) for s in rev_by_seg.index])
        ax.set_ylabel("$ / month")
        st.pyplot(fig)

    st.info(
        "**Revenue at Risk** = Monthly Charges × Churn Probability, summed across customers. "
        "This is an *expected-value estimate*, not a guarantee of lost revenue — it reflects "
        "probability-weighted exposure so the retention team can prioritize outreach."
    )


# ========================================================================
# PAGE 2 — CUSTOMER RISK
# ========================================================================
elif page == "Customer Risk":
    st.title("Customer Risk Lookup")
    st.caption("Search for a specific customer to see their churn risk profile.")

    search_id = st.selectbox(
        "Customer ID",
        options=[""] + sorted(merged["customerID"].unique().tolist()),
        index=0,
    )

    if search_id:
        row = merged[merged["customerID"] == search_id].iloc[0]

        segment_color = {"High Risk": DANGER, "Medium Risk": WARN, "Low Risk": OK}.get(row["RiskSegment"], PRIMARY)
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{row['ChurnProbability']:.1%}")
        c2.markdown(f"**Risk Segment**  \n<span style='color:{segment_color}; font-size:1.4rem; font-weight:600'>{row['RiskSegment']}</span>",
                     unsafe_allow_html=True)
        c3.metric("Monthly Revenue at Risk", f"${row['MonthlyRevenueAtRisk']:.2f}")

        st.subheader("Why this customer is at risk")
        st.write(row["TopRiskFactors"])
        st.caption(f"Explanation method: {row['ExplanationMethod']}")

        st.subheader("Recommended Action")
        st.success(row["SuggestedAction"])

        with st.expander("Full customer record"):
            st.dataframe(row.to_frame().T, use_container_width=True)
    else:
        st.write("Select a Customer ID above to view their risk profile.")


# ========================================================================
# PAGE 3 — CHURN ANALYTICS
# ========================================================================
elif page == "Churn Analytics":
    st.title("Churn Analytics")
    st.caption("Segment-level churn patterns across the customer base.")

    tab1, tab2, tab3, tab4 = st.tabs(["By Contract", "By Tenure", "By Payment Method", "Top Churn Drivers"])

    with tab1:
        grp = customers.groupby("Contract")["ChurnFlag"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots()
        ax.bar(grp.index, grp.values, color=PRIMARY)
        ax.set_ylabel("Churn rate (%)")
        ax.set_title("Churn Rate by Contract Type")
        st.pyplot(fig)
        st.caption("Month-to-month customers churn at a far higher rate than annual/two-year contract holders — "
                   "the lower switching cost of a month-to-month plan is a key risk driver.")

    with tab2:
        grp = customers.groupby("TenureGroup", observed=True)["ChurnFlag"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots()
        ax.bar(grp.index.astype(str), grp.values, color=PRIMARY)
        ax.set_ylabel("Churn rate (%)")
        ax.set_title("Churn Rate by Tenure Group")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)
        st.caption("Churn risk is heavily front-loaded in the first year — the onboarding window matters most.")

    with tab3:
        grp = customers.groupby("PaymentMethod")["ChurnFlag"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots()
        ax.barh(grp.index, grp.values, color=PRIMARY)
        ax.set_xlabel("Churn rate (%)")
        ax.set_title("Churn Rate by Payment Method")
        st.pyplot(fig)
        st.caption("Electronic check payers churn noticeably more than customers on automatic payment methods.")

    with tab4:
        fi = pd.read_csv("reports/feature_importance.csv").head(10)
        fi["feature"] = fi["feature"].str.replace("num__", "").str.replace("cat__", "")
        fig, ax = plt.subplots()
        ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=PRIMARY)
        ax.set_title("Top 10 Churn Drivers (model feature importance)")
        st.pyplot(fig)
        st.caption("This reflects what the model relies on most to predict churn — it is **not** proof of causation.")


# ========================================================================
# PAGE 4 — PREDICTION (new customer form)
# ========================================================================
elif page == "Prediction":
    st.title("Predict Churn for a New Customer")
    st.caption("Enter customer details to get a churn prediction, probability, and recommended action.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment = st.selectbox("Payment Method",
                                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col3:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        total_charges = tenure * monthly_charges  # best estimate for a not-yet-billed hypothetical customer

        raw_row = pd.DataFrame([{
            "customerID": "NEW-CUSTOMER",
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": str(total_charges),
            "Churn": "No",  # placeholder, unused for prediction
        }])

        cleaned = clean_data(raw_row)
        engineered = engineer_features(cleaned)

        pipe = get_pipeline()
        result = predict_batch(pipe, engineered)
        row = result.iloc[0]

        st.divider()
        segment_color = {"High Risk": DANGER, "Medium Risk": WARN, "Low Risk": OK}.get(row["RiskSegment"], PRIMARY)
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{row['ChurnProbability']:.1%}")
        c2.markdown(f"**Risk Level**  \n<span style='color:{segment_color}; font-size:1.4rem; font-weight:600'>{row['RiskSegment']}</span>",
                     unsafe_allow_html=True)
        c3.metric("Est. Monthly Revenue at Risk", f"${row['MonthlyRevenueAtRisk']:.2f}")

        st.subheader("Why")
        st.write(row["TopRiskFactors"])
        st.subheader("Recommended Retention Action")
        st.success(row["SuggestedAction"])


# ========================================================================
# PAGE 5 — MODEL EVALUATION
# ========================================================================
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.caption("Machine learning performance of the deployed churn model, in plain-language business terms.")

    report = get_model_report()
    m = report["final_test_metrics"]

    st.subheader(f"Selected Model: {report['selected_model']}")
    st.write(report["selection_rationale"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.1%}", help="Of all predictions, how many were correct.")
    c2.metric("Precision", f"{m['precision']:.1%}", help="Of customers flagged as churn risks, how many actually churn.")
    c3.metric("Recall", f"{m['recall']:.1%}", help="Of customers who actually churn, how many the model catches.")
    c4.metric("F1 Score", f"{m['f1_score']:.3f}", help="Balance between precision and recall.")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}", help="Overall ability to rank churners above non-churners (1.0 = perfect, 0.5 = random).")

    st.divider()
    col1, col2, col3 = st.columns(3)
    if os.path.exists("reports/figures/confusion_matrix.png"):
        col1.image("reports/figures/confusion_matrix.png", caption="Confusion Matrix")
    if os.path.exists("reports/figures/roc_curve.png"):
        col2.image("reports/figures/roc_curve.png", caption="ROC Curve")
    if os.path.exists("reports/figures/pr_curve.png"):
        col3.image("reports/figures/pr_curve.png", caption="Precision-Recall Curve")

    st.subheader("Model Comparison (all candidates tested)")
    comp_df = pd.DataFrame(report["model_comparison"]).T
    st.dataframe(comp_df, use_container_width=True)
