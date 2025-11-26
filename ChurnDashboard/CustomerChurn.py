#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 09:11:29 2025
"""
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
import seaborn as sb
import joblib
from datetime import datetime
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, roc_auc_score, PrecisionRecallDisplay
from sklearn.ensemble import RandomForestClassifier

# Read in Excel file with all sheets
data = pd.read_excel("Customer_Churn_Data_Large.xlsx", sheet_name=None)

# Split each excel sheet into individual dataframe
cust_demo = data['Customer_Demographics']
trans_hist = data['Transaction_History']
cust_serv = data['Customer_Service']
online_act = data['Online_Activity']
churn_stat = data['Churn_Status']

# Check each sheet for missing values
cust_demo.isnull().sum()
trans_hist.isnull().sum()
cust_serv.isnull().sum()
online_act.isnull().sum()
churn_stat.isnull().sum()

# Look at descriptive information about each dataset's numerical values
cust_demo.describe()
trans_hist.describe()
cust_serv.describe()
online_act.describe()
churn_stat.describe()

plot.hist(cust_demo["Age"])
plot.hist(cust_demo["IncomeLevel"], bins=3)

# Function to introduce random mmissing data
def introduce_missing(data, columns, frac_missing=0.1):
    df_missing = data.copy()
    for col in columns:
        mask = np.random.rand(len(df_missing)) < frac_missing
        df_missing.loc[mask, col] = np.nan
    return df_missing

# Introduce 10% random missing data in Age and Income
cust_demo = introduce_missing(cust_demo, ['Age', 'IncomeLevel'], frac_missing=0.1)
print("\nData with simulated missing values:")
print(cust_demo.head(10))
print("\nMissing values count:")
print(cust_demo.isnull().sum())

# Numeric columns filling in Nulls, For categorical would look to take the mode selection of the sample
cust_demo['Age'] = cust_demo['Age'].fillna(cust_demo['Age'].mean())
cust_demo['IncomeLevel'] = cust_demo['IncomeLevel'].fillna(cust_demo['IncomeLevel'].mode()[0])

# Checking to see if dataset is no longer missing data
# NOTE: For missing IDs that would require more data searching to see what is missing rather than randomly assigning an ID
print(cust_demo.isnull().sum())

# Creating 1 row per customer dataset to make prevent massive nulls when merging data
# Grouping by amount spent and product category for revenue purposes
trans_hist_grouped = trans_hist.groupby(['CustomerID', 'ProductCategory'])['AmountSpent'].sum().reset_index()

# Customer total spend for each customer to generate one row per customer
summary_by_id = trans_hist.groupby('CustomerID').agg(transaction_count=('TransactionID', 'size'),total_spend=('AmountSpent', 'sum')).reset_index()

# Taking latest customer interaction for merged datset
cust_serv_id = cust_serv.drop_duplicates(subset=["CustomerID"], keep="first")

# Merge dataset on customerID
merged_df = cust_demo.merge(summary_by_id, on='CustomerID', how='left') \
                        .merge(cust_serv_id, on='CustomerID', how='left') \
                        .merge(online_act, on='CustomerID', how='left') \
                        .merge(churn_stat, on='CustomerID', how='left')

# Checking that the merge worked correctly, total rows, nulls and if theres duplicate customers
merged_df.info()
merged_df.head()
merged_df.isnull().sum()
merged_df['CustomerID'].duplicated().sum()

# Checking correlation in form of heatmap for the numerical variables of the dataset
corr = merged_df.corr(numeric_only = True)
sb.heatmap(corr, annot=True, cmap="coolwarm")


# Changing dates to numerical datapoints based on latest activity
merged_df['LastLoginDate'] = pd.to_datetime(merged_df['LastLoginDate'])
merged_df['InteractionDate'] = pd.to_datetime(merged_df['InteractionDate'])
today = datetime.now()
merged_df['DaysSinceLastLogin'] = (today - merged_df['LastLoginDate']).dt.days
merged_df['DaysSinceLastInteraction'] = (today - merged_df['InteractionDate']).dt.days

# Distribution plots of variables compared to churn
sb.boxplot(data=merged_df, x="ChurnStatus", y="total_spend")
sb.boxplot(data=merged_df, x="ChurnStatus", y="LoginFrequency")
sb.countplot(data=merged_df, x="Gender", hue="ChurnStatus")

# Splitting variables into numerical and cateogrical To check non-linear relationships between variables and churn 
numerical = ['Age', 'transaction_count', 'total_spend', 'LoginFrequency', 'DaysSinceLastLogin', 'DaysSinceLastInteraction']
categorical = ['Gender', 'MaritalStatus', 'IncomeLevel', 'InteractionType', 'ResolutionStatus', 'ServiceUsage']
target = 'ChurnStatus'

# Create empty lists to store non-linear relationship results
point_bi_results = []
chi_square_results = []

# Use a point biserial r to determine significant numerical variables
for y in numerical:
    p = pointbiserialr(merged_df[target], merged_df[y])
    point_bi_results.append({'Feature': y, 'p-value': p})
    
# Use a chi squared test to determine signicant categorical variables
for x in categorical:
    contingency = pd.crosstab(merged_df[x], merged_df[target])
    chi2, p, dof, expected = chi2_contingency(contingency)
    chi_square_results.append({'Feature': x, 'Chi2': chi2, 'p-value': p})

# Print results, if p > 0.05 then there is no significant relationship with churn
point_bi_df = pd.DataFrame(point_bi_results)
print(point_bi_df)
chi_square_df = pd.DataFrame(chi_square_results)
print(chi_square_df)

# Mapping categorical variables to numerical for both ordered and non-ordered values
income_map = {'Low': 1, 'Medium': 2, 'High': 3}
merged_df['IncomeLevel'] = merged_df['IncomeLevel'].map(income_map)
df_encoded = pd.get_dummies(merged_df, columns=['Gender', 'MaritalStatus', 'ServiceUsage', 'InteractionType', 'ResolutionStatus'], drop_first=True)

# Drop columns that are no longer needed for analysis
df_encoded = df_encoded.drop(columns={'InteractionID', 'InteractionDate', 'LastLoginDate'})
# Fill customers without any interaction with a low value
df_encoded['DaysSinceLastInteraction'] = df_encoded['DaysSinceLastInteraction'].fillna(-1)

# Scale numerical columns so large values do not skew model results
scaler = StandardScaler()
df_encoded[numerical] = scaler.fit_transform(df_encoded[numerical])


# Create x and y for model fitting and splitting data
x = df_encoded.drop(columns=['ChurnStatus', 'CustomerID'])
y = df_encoded['ChurnStatus']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Fit a logistic regression model to the data set since churn is a binary vvarible
model = LogisticRegression(max_iter=1000) # Baseline regression model for comparison
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print the results of the model to determine if it was a good fit or not
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
RocCurveDisplay.from_estimator(model, X_test, y_test)

# If the model is not a good fit try a random forest model that is less dependent on a non-linear relationship between variables
rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=5, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train) # Tree model for non-linear relationships
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Print results of random forest model to see if it fits the data
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
RocCurveDisplay.from_estimator(rf, X_test, y_test)
auc = roc_auc_score(y_test, y_proba)
print("AUC:", auc)

# Plot features to see which features contribute to predicting churn
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_train.columns
plot.figure(figsize=(10,6))
plot.bar(range(len(importances)), importances[indices])
plot.xticks(range(len(importances)), features[indices], rotation=90)
plot.title("Feature Importance (Random Forest)")
plot.tight_layout()
plot.show()

# Redo random forest model with the features that contribute to the churn prediction with same procedure as above
x_important = df_encoded.drop(columns=['ChurnStatus', 'CustomerID', 'Gender_M', 
                                       'ServiceUsage_Online Banking', 'MaritalStatus_Married', 'ServiceUsage_Website', 
                                       'ResolutionStatus_Unresolved', 'MaritalStatus_Single', 'MaritalStatus_Widowed', 
                                       'InteractionType_Inquiry','InteractionType_Feedback'])
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_important, y, test_size=0.2, random_state=42)
rf2 = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=5, class_weight="balanced", random_state=42)
rf2.fit(X_train2, y_train2) # Tree model focusing on only significant features for non-linear relationships
y_pred2 = rf2.predict(X_test2)
y_proba2 = rf2.predict_proba(X_test2)[:, 1]
# Plot results
print(classification_report(y_test2, y_pred2))
print(confusion_matrix(y_test2, y_pred2))
RocCurveDisplay.from_estimator(rf2, X_test2, y_test2)
auc2 = roc_auc_score(y_test2, y_proba2)
print("AUC:", auc2)

# Show feature relationships
importances2 = rf2.feature_importances_
indices2 = np.argsort(importances2)[::-1]
features2 = X_train2.columns
plot.figure(figsize=(10,6))
plot.bar(range(len(importances2)), importances2[indices2])
plot.xticks(range(len(importances2)), features2[indices2], rotation=90)
plot.title("Feature Importance (Random Forest)")
plot.tight_layout()
plot.show()

# Plot of recall from random forest model
PrecisionRecallDisplay.from_estimator(rf2, X_test2, y_test2)

# Adjust decision threshold to work better with normal churn probabilities
y_pred30 = (y_proba2 >= 0.30).astype(int)
print(confusion_matrix(y_test2, y_pred30))
print(classification_report(y_test2, y_pred30))


# =============================================================================
# Key Drivers of Churn (from Random Forest Importance):
# 1. DaysSinceLastLogin — customers who stop logging in are at far higher churn risk.
# 2. total_spend — lower spend correlates with churn (revenue risk!)
# 3. LoginFrequency — infrequent users tend to churn.
# 4. DaysSinceLastInteraction — long unresolved service gaps increase churn likelihood.
# 
# Demographic variables contribute little to prediction, indicating churn is behavior-driven rather than profile-driven.
# =============================================================================

# Final summary of model with threshold
print("Final Model Performance (Random Forest, threshold=0.30):")
print("AUC:", auc2)
print(confusion_matrix(y_test2, y_pred30))
print(classification_report(y_test2, y_pred30))

# Save model and datasets for dashboard use
model = rf2
joblib.dump(model, 'saved_model.pkl')
df_encoded.to_parquet("model_ready_data.parquet", engine="fastparquet")
merged_df.to_parquet("merged_data.parquet", engine="fastparquet")

