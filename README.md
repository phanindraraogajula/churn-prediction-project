# 📉 Customer Churn Prediction — Machine Learning Analysis

## 📌 Project Overview
This project predicts customer churn using a Random Forest ML model, identifies the top drivers of churn, and provides actionable marketing recommendations to improve customer retention.

## 🎯 Business Problem
Acquiring a new customer costs 5x more than retaining an existing one. This analysis answers:
- Which customers are most likely to churn?
- What are the biggest drivers of churn?
- How can marketing teams intervene before it's too late?

## 📊 Key Results
| Metric | Value |
|--------|-------|
| Model Accuracy | 64% |
| ROC-AUC Score | 0.70 |
| Total Customers | 10,000 |
| Overall Churn Rate | 47.1% |
| Gold Member Churn | 31.8% vs 51% Bronze |

## 🔑 Top Churn Drivers
1. Last Login Days (21.8%) — Inactive customers churn most
2. Monthly Spend (13.1%) — Low spenders are high risk
3. Email Open Rate (12.2%) — Disengaged customers leave
4. Number of Purchases (11.9%) — Low purchase frequency = high churn
5. Tenure Months (11.8%) — New customers churn faster

## 💡 Business Recommendations
- **Win-back campaign** for customers inactive 90+ days
- **Loyalty upgrades** — move Bronze to Silver to cut churn by 20%
- **Email re-engagement** for customers with open rate below 20%
- **Onboarding program** for customers in first 6 months

## 🔧 Tools & Technologies
- Python (Pandas, NumPy, Sklearn, Matplotlib, Seaborn)
- Machine Learning (Random Forest, Logistic Regression)
- ROC-AUC, Confusion Matrix, Feature Importance
- GitHub

## 🧠 Methodology
1. Generated realistic dataset of 10,000 customers
2. Engineered churn probability based on real business rules
3. Built Random Forest classifier (100 estimators)
4. Evaluated model with ROC-AUC, precision, recall
5. Identified top churn drivers via feature importance
6. Segmented churn by membership tier and channel
7. Delivered actionable marketing recommendations

## 👤 Author
**Phanindra Gajula**
- 🔗 [LinkedIn](https://www.linkedin.com/in/phanindraraogajula)
- 🔗 [GitHub](https://github.com/phanindraraogajula)
