import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

df = pd.DataFrame({
    'customer_id': range(1, n + 1),
    'age': np.random.randint(18, 70, n),
    'gender': np.random.choice(['Male', 'Female'], n),
    'tenure_months': np.random.randint(1, 60, n),
    'monthly_spend': np.round(np.random.uniform(20, 200, n), 2),
    'num_purchases': np.random.randint(1, 50, n),
    'support_tickets': np.random.randint(0, 10, n),
    'email_open_rate': np.round(np.random.uniform(0, 1, n), 2),
    'last_login_days': np.random.randint(1, 180, n),
    'preferred_channel': np.random.choice(['Email', 'Social', 'Search', 'Direct'], n),
    'membership_tier': np.random.choice(['Bronze', 'Silver', 'Gold'], n, p=[0.5, 0.3, 0.2]),
})

# Churn logic — realistic business rules
churn_prob = (
    0.3
    + (df['last_login_days'] > 90) * 0.3
    + (df['support_tickets'] > 5) * 0.2
    + (df['tenure_months'] < 6) * 0.2
    + (df['email_open_rate'] < 0.2) * 0.1
    - (df['membership_tier'] == 'Gold') * 0.2
    - (df['num_purchases'] > 30) * 0.15
)
churn_prob = churn_prob.clip(0, 1)
df['churned'] = np.random.binomial(1, churn_prob)

df.to_csv('churn_data.csv', index=False)
print("Dataset created! Shape:", df.shape)
print("Churn Rate:", round(df['churned'].mean() * 100, 2), "%")
print(df.head())
