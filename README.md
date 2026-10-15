# Credit Risk Model — German Credit Data

Predicts whether a loan applicant is a good or bad credit risk from their profile, and checks how far the model can be trusted. It follows my earlier [Credit Risk Exposure Dashboard](https://public.tableau.com/app/profile/vasavi.chinnala/viz/Credit-Risk-Exposure-Dashboard/Dashboard1) in Tableau, which describes the same applicants.

## Data
- 1,000 loan applicants, about 30% labelled bad risk
- Source: "German Credit Risk – With Target" on Kaggle (based on the UCI German Credit data)
- File: `german_credit_data.csv`

## Approach
1. Cleaned the data. Missing savings/checking account values were labelled `unknown` instead of being dropped.
2. Explored bad-risk rates by account status and credit amount.
3. Excluded `Sex` from the features for fairness reasons, and tested the effect of removing `Age`.
4. Trained three models on a stratified 75/25 train/test split: a baseline, logistic regression, and a decision tree (depth 4).
5. Validated with accuracy, precision, recall and confusion matrices, then looked at which features drive each model.

## Results (test set)
| Model | Accuracy | Precision (bad) | Recall (bad) | F1 (bad) |
|---|---|---|---|---|
| Baseline (always "good") | 0.700 | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.688 | 0.486 | 0.720 | 0.581 |
| Decision tree | 0.656 | 0.456 | 0.760 | 0.570 |

- Accuracy alone is misleading: the baseline scores 70% without catching a single bad applicant.
- The models catch about three-quarters of risky applicants, at the cost of many false alarms (precision under 50%).
- Applicants with no checking account on file had the lowest bad-risk rate (about 12%, against about 49% for "little"). The data shows this pattern but cannot explain it.
- Removing `Age` changed logistic-regression recall from 0.72 to 0.69, a small change on a test set of about 75 bad applicants.

## Limitations
- Only 1,000 rows, so results shift with a different train/test split.
- The data is old and from one country, so this is a learning project, not a real lending model.
- The costs of wrongly approving a bad loan and wrongly rejecting a good one are not modelled.
- No cross-validation or hyperparameter tuning yet.

## How to run
Open `Credit_Risk_Model_German_Credit.ipynb` in Google Colab, upload `german_credit_data.csv` when asked, and run the cells in order.

Tools: Python, pandas, scikit-learn, matplotlib
