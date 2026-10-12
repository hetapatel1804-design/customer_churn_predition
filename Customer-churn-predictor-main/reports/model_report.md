# Model Report — Customer Churn Prediction

## Data Quality Summary

- **n_rows**: 7043
- **n_cols**: 21
- **duplicate_customer_ids**: 0
- **duplicate_rows**: 0
- **missing_values_per_column**: {'customerID': 0, 'gender': 0, 'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 0, 'tenure': 0, 'PhoneService': 0, 'MultipleLines': 0, 'InternetService': 0, 'OnlineSecurity': 0, 'OnlineBackup': 0, 'DeviceProtection': 0, 'TechSupport': 0, 'StreamingTV': 0, 'StreamingMovies': 0, 'Contract': 0, 'PaperlessBilling': 0, 'PaymentMethod': 0, 'MonthlyCharges': 0, 'TotalCharges': 0, 'Churn': 0}
- **total_charges_blank_strings**: 11
- **churn_value_counts**: {'No': 5174, 'Yes': 1869}
- **churn_rate_pct**: 26.54

## Model Comparison (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC (mean±std) |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.7367 | 0.5026 | 0.7888 | 0.6139 | 0.8424 | 0.8457 ± 0.0113 |
| DecisionTree | 0.7253 | 0.489 | 0.7701 | 0.5981 | 0.7944 | 0.7853 ± 0.0039 |
| RandomForest | 0.7757 | 0.6 | 0.4652 | 0.5241 | 0.8182 | 0.8237 ± 0.0106 |
| GradientBoosting | 0.802 | 0.6644 | 0.5134 | 0.5792 | 0.8433 | 0.848 ± 0.0121 |

## Selected Model: **LogisticRegression**

Weighted score = 0.6*ROC-AUC + 0.4*Recall. A missed churner (false negative) costs the business a lost customer with no retention attempt, while a false positive only costs an unnecessary outreach/discount to a customer who would have stayed anyway — so recall and ranking quality (ROC-AUC) are weighted above raw accuracy.

## Hyperparameter Tuning

Best params: `{'classifier__solver': 'liblinear', 'classifier__C': 0.01}`

Best CV ROC-AUC during tuning: **0.8461**

## Final Test-Set Performance (tuned model)

- **accuracy**: 0.736
- **precision**: 0.5017
- **recall**: 0.7914
- **f1_score**: 0.6141
- **roc_auc**: 0.8404

```
              precision    recall  f1-score   support

    No Churn       0.90      0.72      0.80      1035
       Churn       0.50      0.79      0.61       374

    accuracy                           0.74      1409
   macro avg       0.70      0.75      0.71      1409
weighted avg       0.80      0.74      0.75      1409

```

## Top Churn Drivers

| feature                              |   importance |
|:-------------------------------------|-------------:|
| num__ContractRiskScore               |     0.431651 |
| num__tenure                          |     0.409797 |
| num__MonthlyCharges                  |     0.297127 |
| cat__InternetService_Fiber optic     |     0.23413  |
| cat__TenureGroup_0-12 (New)          |     0.223912 |
| cat__InternetService_DSL             |     0.223892 |
| cat__PaperlessBilling_No             |     0.197368 |
| cat__Contract_Two year               |     0.180565 |
| cat__MultipleLines_No                |     0.165691 |
| num__IsElectronicCheck               |     0.152146 |
| cat__Contract_Month-to-month         |     0.149429 |
| cat__TenureGroup_25-48 (Established) |     0.147977 |
| cat__PhoneService_Yes                |     0.139241 |
| cat__Dependents_Yes                  |     0.132965 |
| num__HasOnlineSecurity               |     0.125299 |

*Note: this is model feature importance, not proof of causation. It shows which features the model relies on most to make predictions, which is correlational within the model, not a causal claim.*
