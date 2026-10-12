# Resume Description & Interview Guide

## Resume-ready project description

**Customer Churn Prediction System** | Python, Scikit-learn, Pandas, SQL, Streamlit
*Personal project · [GitHub link]*

- Built an end-to-end churn prediction system on the IBM Telco Customer Churn dataset (7,043 customers), including a leakage-safe preprocessing pipeline, feature engineering, and business-informed model selection.
- Compared 4 classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) with stratified cross-validation; selected the final model using a business-weighted score (ROC-AUC + Recall) rather than raw accuracy, since missing an actual churner is costlier than a false alarm — achieved 0.84 ROC-AUC and 79% recall on held-out test data.
- Engineered 8 business-driven features (tenure lifecycle stage, contract risk score, service engagement count, etc.) and validated them against target leakage.
- Built a customer-level explainability layer using exact linear-model coefficient contributions, translating each prediction into plain-language risk factors and a rule-based recommended retention action.
- Designed a SQL analysis layer (10+ queries incl. CTEs and window functions) and a 5-page Streamlit dashboard (executive KPIs, customer lookup, analytics, live prediction form, model evaluation) surfacing ~$213K/month in probability-weighted revenue at risk.

*(Trim to 3-4 bullets depending on resume length; the ROC-AUC/recall numbers and the $213K figure are the two strongest, most concrete claims — lead with those if space is tight.)*

---

## How to explain this project in an interview (2-minute version)

"I built a churn prediction system on a real telecom dataset — about 7,000 customers, 26.5% of whom had churned. The interesting part wasn't really the modeling — logistic regression, random forest, and gradient boosting all landed in a similar performance range — it was the decisions around it.

First, I found a real data quality bug: `TotalCharges` was stored as text and had 11 blank values that pandas' `.isnull()` didn't catch, because they were empty strings, not `NaN`. I traced those back to customers with zero tenure — brand new accounts that hadn't been billed yet — and handled them as structurally zero rather than imputing a mean, which would have been wrong.

Second, model selection: I didn't just pick the highest-accuracy model. For churn, a false negative — telling the business a customer is safe when they're actually about to leave — is much more expensive than a false positive, since the retention team simply never reaches out. So I picked based on a weighted score of ROC-AUC and recall, which is actually why I ended up choosing logistic regression over gradient boosting, even though gradient boosting had higher raw accuracy — logistic regression caught 79% of actual churners versus gradient boosting's 51%.

Third, I made the output business-usable, not just a probability number: every customer gets a risk segment, an explanation for why they're at risk — using the actual model coefficients, not a black box — and a suggested retention action, plus an estimate of monthly revenue at risk so the team can prioritize."

---

## Common interview questions and honest answers

**Q: Why did you choose Logistic Regression as the final model when Gradient Boosting had higher accuracy?**
A: Accuracy treats a missed churner and a false alarm as equally costly, but they're not — a missed churner is a guaranteed lost customer with no retention attempt, while a false alarm just costs an unnecessary outreach. I scored models on `0.6×ROC-AUC + 0.4×Recall` instead, and Logistic Regression won on both: 0.8424 ROC-AUC vs Gradient Boosting's 0.8433 (essentially tied) but 78.9% recall vs Gradient Boosting's 51.3%. Given the business cost asymmetry, that recall gap decided it.

**Q: How did you avoid data leakage?**
A: Two ways specifically: I split into train/test *before* fitting any preprocessing (imputers, scaler, one-hot encoder are all fit only on the training fold, inside a single sklearn `Pipeline`), and I excluded `customerID` from the model features entirely, since it's a unique identifier with no real predictive signal and would just let the model memorize rows.

**Q: How would you improve this given more time/resources?**
A: SHAP wasn't installable in the sandbox I built this in, so I implemented exact coefficient-based local explanations for the linear model instead — I'd swap in SHAP for a model-agnostic, more standard explanation format. I'd also calibrate the predicted probabilities (`CalibratedClassifierCV`) before trusting them as literal probabilities for the revenue-at-risk calculation, and validate the 0.70/0.40 risk thresholds against actual retention-offer costs rather than using them as fixed defaults.

**Q: Precision is only ~50% — is that a problem?**
A: It means about half of customers flagged "high risk" don't actually churn. Whether that's acceptable depends on the cost of a false positive — here, that's typically a discount offer or a support call, which is cheap relative to losing a customer entirely. That's exactly why I optimized for recall over precision in this business context; a different business (e.g., where the "action" was an expensive intervention) might reasonably prefer a higher-precision model even at the cost of recall.

**Q: Does high feature importance mean that feature causes churn?**
A: No — importance tells you what the model relies on to make accurate predictions, which is a statement about the model, not the world. `TotalCharges` and `tenure` are highly collinear, for instance, so how importance splits between them is partly an artifact of the model, not proof that one causes churn more than the other. I called this out explicitly in the notebooks and README rather than overclaiming.

**Q: How would you deploy this in production?**
A: The saved artifact is a single `joblib` file containing the *entire* pipeline — preprocessing and model together — so there's no risk of training/inference mismatch. I'd wrap `predict.py`'s `predict_batch()` function in a FastAPI endpoint, containerize it, and schedule a batch scoring job (e.g., nightly) that refreshes `scored_customers.csv` / the database table the dashboard reads from. I'd also add data drift monitoring on the input feature distributions, since telecom pricing and contract mixes shift over time.
