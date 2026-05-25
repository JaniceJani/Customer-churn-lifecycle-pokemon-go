"""
Data-Driven (Customer) Insights 2025-2026
Pokémon Go Case Assignment
"""
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf # Using this for our logistic regression
import matplotlib.pyplot as plt
from plotnine import *
import stepwise as stepwise
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, confusion_matrix
import os 
import datetime
import seaborn as sns 
# ==========================================
# 1. READ IN DATAFRAMES
# ==========================================
# Grabbing the exact folder where this script is saved so Python always finds the CSVs
script_dir = os.path.dirname(os.path.abspath(__file__))
summerfintrx = pd.read_csv(os.path.join(script_dir, 'summerfintrx.csv'), parse_dates=['Date'])
summersesstrx = pd.read_csv(os.path.join(script_dir, 'summersesstrx.csv'), parse_dates=['Date'])
fallfintrx = pd.read_csv(os.path.join(script_dir, 'fallfintrx.csv'), parse_dates=['Date'])
fallsesstrx = pd.read_csv(os.path.join(script_dir, 'fallsesstrx.csv'), parse_dates=['Date'])
customerdata = pd.read_csv(os.path.join(script_dir, 'customerdata.csv'), parse_dates=['Registrationdate'])
# Dropping the time part of the dates the "Pandas Way" using normalize()
summerfintrx['Date'] = pd.to_datetime(summerfintrx['Date']).dt.normalize()
summersesstrx['Date'] = pd.to_datetime(summersesstrx['Date']).dt.normalize()
fallfintrx['Date'] = pd.to_datetime(fallfintrx['Date']).dt.normalize()
fallsesstrx['Date'] = pd.to_datetime(fallsesstrx['Date']).dt.normalize()
customerdata['Registrationdate'] = pd.to_datetime(customerdata['Registrationdate']).dt.normalize()
# ==========================================
# QUESTION 1: BASETABLE & PROFILING
# ==========================================
print("\n--- Q1: Basetable Creation & Profiling ---")
# 1. Identify active summer customers (At least 1 play session)
active_customers = summersesstrx['CustomerID'].unique()
basetable = customerdata[customerdata['CustomerID'].isin(active_customers)].copy()
# 2. Aggregate Usage Data (Play Sessions)
usage_agg = summersesstrx.groupby('CustomerID').agg(
    Total_Sessions=('SessionID', 'count'),
    Total_Distance=('Distance', 'sum'),
    Total_Duration=('Duration', 'sum'),
    Total_Experience=('Experience', 'sum'),
    Last_Play_Date=('Date', 'max') # For Recency
).reset_index()
# 3. Aggregate Financial Data (Microtransactions)
# Note: We use summerfintrx. If a user isn't in here, they spent 0.
financial_agg = summerfintrx.groupby('CustomerID').agg(
    Total_Purchases=('TransactionID', 'count'),
    Total_Spend=('Value', 'sum'),
    Last_Purchase_Date=('Date', 'max') # For Recency
).reset_index()
# 4. Merge into the Final Basetable
basetable = pd.merge(basetable, usage_agg, on='CustomerID', how='left')
basetable = pd.merge(basetable, financial_agg, on='CustomerID', how='left')
# Fill NaNs for non-spenders
basetable['Total_Purchases'] = basetable['Total_Purchases'].fillna(0)
basetable['Total_Spend'] = basetable['Total_Spend'].fillna(0)
# Calculate Recency (Days since last action, assuming end of summer is 2022-08-31)
end_of_summer = pd.to_datetime('2022-08-31')
# Force the columns back to pandas datetime objects to prevent the .dt accessor error
basetable['Last_Play_Date'] = pd.to_datetime(basetable['Last_Play_Date'])
basetable['Last_Purchase_Date'] = pd.to_datetime(basetable['Last_Purchase_Date'])
basetable['Recency_Play'] = (end_of_summer - basetable['Last_Play_Date']).dt.days
basetable['Recency_Purchase'] = (end_of_summer - basetable['Last_Purchase_Date']).dt.days
# Fill missing purchase recency with a large number (e.g., 999) or drop for non-spenders later
basetable['Recency_Purchase'] = basetable['Recency_Purchase'].fillna(999)
# Print General Profile Stats for the Report
print(f"Total Active Players: {len(basetable)}")
print(f"Average Age: {basetable['Age'].mean():.1f}")
print(f"Gender Split:\n{basetable['Gender'].value_counts(normalize=True)*100}")
print(f"Income Distribution:\n{basetable['Income'].value_counts(normalize=True)*100}")
print(f"Average Sessions: {basetable['Total_Sessions'].mean():.2f}")
print(f"Average Distance Walked: {basetable['Total_Distance'].mean():.2f} km")
# Calculate Fall Boost Promo percentage
promo_pct = basetable['fallbonus'].mean() * 100
print(f"Received Fall Boost Promo: {promo_pct:.1f}%")
# Create the spenders dataframe before calculating CLV
spenders = basetable[basetable['Total_Spend'] > 0].copy()
try:
from lifetimes import BetaGeoFitter, GammaGammaFitter
# Lifetimes requires specific formatting: Frequency, Recency, T (Tenure), and Monetary Value
    clv_data = spenders[['CustomerID', 'Total_Purchases', 'Recency_Purchase', 'Total_Spend']].copy()
    clv_data.columns = ['CustomerID', 'frequency', 'recency', 'monetary_value']
    clv_data['T'] = 122 # Total days in the summer period
# We only fit models on returning customers (frequency > 0)
    clv_data = clv_data[clv_data['frequency'] > 0]
if len(clv_data) > 0:
# ADDED PENALIZER: 0.1 forces the math to converge despite the highly skewed gaming data
        bgf 
= BetaGeoFitter(penalizer_coef=0.1)
        bgf.fit(clv_data['frequency'], clv_data['recency'], clv_data['T'])
# ADDED PENALIZER: 0.1 for Gamma-Gamma as well
        ggf 
= GammaGammaFitter(penalizer_coef=0.1)
        ggf.fit(clv_data['frequency'], clv_data['monetary_value'])
# Calculate CLV (12 months, 10% discount rate)
        clv_data['CLV'] = ggf.customer_lifetime_value(
            bgf,
            clv_data['frequency'],
            clv_data['recency'],
            clv_data['T'],
            clv_data['monetary_value'],
            time=12, # months
            discount_rate=0.01 # roughly 10% annually adjusted for monthly
        )
print(f"\nAverage Calculated CLV for Spenders: €{clv_data['CLV'].mean():.2f}")
        type_map_clean = {1: 'Walker', 2: 'Miscellaneous', 3: 'Social Raider', 4: 'Catcher'}
        clv_data = clv_data.merge(basetable[['CustomerID', 'CustomerType']], on='CustomerID', how='left')
        clv_data['Segment'] = clv_data['CustomerType'].map(type_map_clean)
print("\n--- CLV Summary Statistics (Spenders Only) ---")
print(clv_data['CLV'].describe().round(2))
print("\n--- Top 10 Customers by Predicted CLV ---")
        top_10_clv = clv_data[['CustomerID', 'Segment', 'monetary_value', 'CLV']].sort_values(by='CLV', ascending=False).head(10)
print(top_10_clv.round(2))
except ImportError:
print("\nSkipping advanced CLV calculation: 'lifetimes' package not installed. Run 'pip install lifetimes'.")    
# --- Q1 Extension: Profiling the 4 Customer Types --
print("\n--- Profiling the 4 Customer Segments ---")
# 1=Walker, 2=Misc, 3=Social Raider, 4=Catcher
type_map = {1: '1-Walker', 2: '2-Misc', 3: '3-Social Raider', 4: '4-Catcher'}
basetable['Segment'] = basetable['CustomerType'].map(type_map)
# Let's see how their gameplay actually differs!
segment_profile = basetable.groupby('Segment')[['Total_Distance', 'Total_Sessions', 'Total_Spend']].mean().round(2)
print(segment_profile)
# ==========================================
# QUESTION 2: LIFECYCLE GRIDS (Summer)
# ==========================================
print("\n--- Q2: Lifecycle Grids ---")
# 1. Financial Grid (Spenders only)
spenders = basetable[basetable['Total_Spend'] > 0].copy()
if len(spenders) > 2:
# Since max purchases is only 5, statistical qcut fails. We use manual cuts instead.
    spenders['Freq_Tier'] = pd.cut(spenders['Total_Purchases'], bins=[0, 1, 2, 100], labels=['1 Purchase', '2 Purchases', '3+ Purchases'])
    spenders['Recency_Tier'] = pd.qcut(spenders['Recency_Purchase'], q=3, duplicates='drop')
    financial_grid = pd.crosstab(spenders['Recency_Tier'], spenders['Freq_Tier'])
print("\nFinancial Lifecycle Grid (Spenders):")
print(financial_grid)
# 2. Game Usage Grid (All Active Players)
basetable['Play_Freq_Tier'] = pd.qcut(basetable['Total_Sessions'], q=3, duplicates='drop')
basetable['Play_Recency_Tier'] = pd.qcut(basetable['Recency_Play'], q=3, duplicates='drop')
usage_grid = pd.crosstab(basetable['Play_Recency_Tier'], basetable['Play_Freq_Tier'])
print("\nGame Usage Lifecycle Grid (All Active Players):")
print(usage_grid)
# ==========================================
# QUESTION 3: CHURN ANALYSIS (Fall)
# ==========================================
print("\n--- Q3: Churn Analysis & Logistic Regression ---")
# Target Cohort: Paid in summer
summer_payers = basetable[basetable['Total_Spend'] > 0].copy()
# Define the price map for fall transactions
price_map = {1: 2.99, 2: 4.99, 3: 9.99, 4: 25.0, 5: 99.0, 0: 0.0}
# Checking fall revenue
fallfintrx['Revenue_Fall'] = fallfintrx['ProductID'].map(price_map)
fall_spenders = fallfintrx.groupby('CustomerID').agg(Monetary_Fall=('Revenue_Fall', 'sum')).reset_index()
churn_df = summer_payers.merge(fall_spenders, on='CustomerID', how='left')
churn_df['Monetary_Fall'] = churn_df['Monetary_Fall'].fillna(0) 
# Churn Target: 1 = Left, 0 = Stayed
churn_df['Churn'] = np.where(churn_df['Monetary_Fall'] == 0, 1, 0)
print(f"Overall Churn Rate: {churn_df['Churn'].mean() * 100:.2f}%\n")
# Prepare Data for Stepwise (Dummifying Categoricals)
model_data = churn_df[['Churn', 'fallbonus', 'CustomerType', 'Recency_Play', 'Total_Duration', 'Total_Distance', 'Age', 'Income', 'Gender']].
copy()
# ADDED dtype=int to force 1s and 0s instead of True/False so statsmodels doesn't crash
model_data = pd.get_dummies(model_data, columns=['CustomerType', 'Income', 'Gender'], drop_first=True, dtype=int)
# Force everything to be a float so statsmodels is perfectly happy
model_data = model_data.astype(float)
model_data = model_data.dropna() 
X = model_data.drop(columns=['Churn'])
y = model_data['Churn']
try:
# Running the Professor's Stepwise File to optimize via AIC
    selected_features, log, final_model = stepwise.forwardSelection(X, y, model_type="logistic", elimination_criteria="aic")
print("\n--- Stepwise Feature Selection Log ---")
print(log)
# Generate Predictions for Evaluation
# Filter out 'intercept' so pandas can slice the dataframe correctly
    features_for_pred = [f for f in selected_features if f != 'intercept']
    model_data['Predicted_Prob'] = final_model.predict(sm.add_constant(X[features_for_pred]))
    model_data['Predicted_Class'] = np.where(model_data['Predicted_Prob'] > 0.5, 1, 0)
# 1. Base Metrics
    auc_score = roc_auc_score(model_data['Churn'], model_data['Predicted_Prob'])
print(f"\nFinal Model Accuracy: {accuracy_score(model_data['Churn'], model_data['Predicted_Class']):.3f}")
print(f"Final Model AUC: {auc_score:.3f}")
# 2. Confusion Matrix
    cm 
= confusion_matrix(model_data['Churn'], model_data['Predicted_Class'])
print("\nConfusion Matrix:")
print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
# 3. Lift Calculation Data (Top 20% targeting)
    lift_df = model_data[['Churn', 'Predicted_Prob']].sort_values(by='Predicted_Prob', ascending=False)
    top_20_cutoff = int(len(lift_df) * 0.2)
    top_20_users = lift_df.head(top_20_cutoff)
    churners_in_top_20 = top_20_users['Churn'].sum()
    total_churners = lift_df['Churn'].sum()
    lift = (churners_in_top_20 / top_20_cutoff) / (total_churners / len(lift_df))
    percent_captured = (churners_in_top_20 / total_churners) * 100
print(f"\nBusiness ROI (Lift):")
print(f"By targeting the riskiest 20% of players flagged by our model, Niantic captures {percent_captured:.1f}% of all actual churners.")
print(f"This is a Lift of {lift:.2f}x compared to random targeting.")
except Exception as e:
print("Model failed:", e)
# Fulfilling the rubric requirement to update the basetable with Churn info
basetable = basetable.merge(churn_df[['CustomerID', 'Churn']], on='CustomerID', how='left')
# ==========================================
# VISUALIZATIONS (Proof of Work)
# ==========================================
print("\nGenerating visual plots...")
# 1. Plotting the ROC Curve 
fpr, tpr, thresholds = roc_curve(model_data['Churn'], model_data['Predicted_Prob'])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Logistic Regression (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.title('ROC Curve: Fall Churn Prediction')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(os.path.join(script_dir, 'ROC_Curve.png'))
plt.close()
# 2. Plotting the Usage Heatmap 
plt.figure(figsize=(8, 6))
sns.heatmap(usage_grid, annot=True, fmt="d", cmap="Blues")
plt.title('Game Usage Lifecycle Grid (All Active Players)')
plt.savefig(os.path.join(script_dir, 'Usage_Heatmap.png'))
plt.close()
print("Visuals saved to your folder as PNG files!")
# ==========================================
# ADVANCED MANAGERIAL GRIDS (Segment & Bonus)
# ==========================================
print("\n--- Advanced Managerial Grids ---")
# Map the Customer Types to their actual names
type_map = {1: '1-Walker', 2: '2-Misc', 3: '3-Social Raider', 4: '4-Catcher'}
churn_df['Segment'] = churn_df['CustomerType'].map(type_map)
# Matrix 1: Customer Segment vs Churn Rate & Spend
segment_matrix = churn_df.groupby('Segment').agg(
    Total_Paying_Players=('CustomerID', 'count'),
    Avg_Summer_Spend=('Total_Spend', 'mean'),
    Churn_Rate_Pct=('Churn', lambda x: x.mean() * 100)
).round(2)
print("\n1. Customer Segment vs Churn Rate Matrix:")
print(segment_matrix)
# Matrix 2: The Fallbonus Effect Matrix
# This shows the exact churn rate for those who got the bonus vs those who didn't
bonus_effect = churn_df.pivot_table(
    index='Segment', 
    columns='fallbonus', 
    values='Churn', 
    aggfunc=lambda x: round(x.mean() * 100, 2)
)
bonus_effect.columns = ['No Bonus (Churn %)', 'Received Bonus (Churn %)']
bonus_effect['Bonus Impact (Lift %)'] = bonus_effect['No Bonus (Churn %)'] - bonus_effect['Received Bonus (Churn %)']
print("\n2. Fallbonus Effect on Churn Rate by Segment:")
print(bonus_effect)
# Exporting for report
basetable.to_csv(os.path.join(script_dir, 'final_basetable.csv'), index=False)
churn_df.to_csv(os.path.join(script_dir, 'churn_analysis_data.csv'), index=False)
print("\nSuccess! Data exported.")