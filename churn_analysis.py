import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('churn_data.csv')

print("=" * 50)
print("CUSTOMER CHURN PREDICTION REPORT")
print("=" * 50)

print("\n📊 Dataset Overview:")
print(f"Total Customers: {len(df)}")
print(f"Churned: {df['churned'].sum()} ({df['churned'].mean()*100:.1f}%)")
print(f"Retained: {(df['churned']==0).sum()} ({(df['churned']==0).mean()*100:.1f}%)")

# --- 1. Feature Engineering ---
le = LabelEncoder()
df['gender_enc'] = le.fit_transform(df['gender'])
df['channel_enc'] = le.fit_transform(df['preferred_channel'])
df['tier_enc'] = le.fit_transform(df['membership_tier'])

features = ['age', 'tenure_months', 'monthly_spend', 'num_purchases',
            'support_tickets', 'email_open_rate', 'last_login_days',
            'gender_enc', 'channel_enc', 'tier_enc']

X = df[features]
y = df['churned']

# --- 2. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Random Forest Model ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("\n🤖 Random Forest Model Performance:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# --- 4. Feature Importance ---
importance_df = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🔑 Top Churn Drivers:")
print(importance_df)

# --- 5. Churn by Segment ---
print("\n📱 Churn Rate by Membership Tier:")
print(df.groupby('membership_tier')['churned'].mean().round(3) * 100)

print("\n📣 Churn Rate by Channel:")
print(df.groupby('preferred_channel')['churned'].mean().round(3) * 100)

# --- 6. Visualizations ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Churn Prediction Dashboard', fontsize=16, fontweight='bold')

# Chart 1 - Feature Importance
axes[0,0].barh(importance_df['feature'], importance_df['importance'], color='#2563EB')
axes[0,0].set_title('Top Churn Drivers (Feature Importance)')
axes[0,0].set_xlabel('Importance Score')
axes[0,0].invert_yaxis()

# Chart 2 - Churn by Membership Tier
tier_churn = df.groupby('membership_tier')['churned'].mean() * 100
axes[0,1].bar(tier_churn.index, tier_churn.values, color=['#CD7F32', '#C0C0C0', '#FFD700'])
axes[0,1].set_title('Churn Rate by Membership Tier (%)')
axes[0,1].set_ylabel('Churn Rate (%)')

# Chart 3 - ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1,0].plot(fpr, tpr, color='#2563EB', label=f'AUC = {roc_auc_score(y_test, y_prob):.3f}')
axes[1,0].plot([0,1], [0,1], 'k--')
axes[1,0].set_title('ROC Curve')
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate')
axes[1,0].legend()

# Chart 4 - Churn by Tenure
df['tenure_group'] = pd.cut(df['tenure_months'], bins=[0,6,12,24,60], labels=['0-6m','6-12m','12-24m','24m+'])
tenure_churn = df.groupby('tenure_group')['churned'].mean() * 100
axes[1,1].bar(tenure_churn.index.astype(str), tenure_churn.values, color='#E74C3C')
axes[1,1].set_title('Churn Rate by Customer Tenure (%)')
axes[1,1].set_ylabel('Churn Rate (%)')

plt.tight_layout()
plt.savefig('churn_charts.png', dpi=150, bbox_inches='tight')
print("\n✅ Charts saved as churn_charts.png")

importance_df.to_csv('churn_feature_importance.csv', index=False)
print("✅ Results saved as churn_feature_importance.csv")
