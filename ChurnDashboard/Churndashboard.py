#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 20:24:24 2025

@author: tholcomb
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score
import os

base = os.path.dirname(__file__)
model_path = os.path.join(base, "saved_model.pkl")
encoded_path = os.path.join(base, "model_ready_data.parquet")
merged_path = os.path.join(base, "merged_data.parquet")

model = joblib.load(model_path)
encoded_df = pd.read_parquet(encoded_path)
merged_df = pd.read_parquet(merged_path)

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

trained_features = model.feature_names_in_
x = encoded_df[trained_features]
y = encoded_df["ChurnStatus"]

# Predict probabilities
encoded_df["ChurnProbability"] = model.predict_proba(x)[:, 1]
merged_df["ChurnProbability"] = model.predict_proba(x)[:, 1]
encoded_df["Prediction"] = model.predict(x)
merged_df["Prediction"] = model.predict(x)

income_map = {1: 'Low', 2: 'Medium', 3:'High'}
churn_map = {0: 'No', 1:'Yes'}
merged_df['IncomeLevel'] = merged_df['IncomeLevel'].map(income_map)
merged_df['Churn Status'] = merged_df['ChurnStatus'].map(churn_map)
# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Filters")
prob_threshold = st.sidebar.slider("Churn Threshold", 0.0, 1.0, 0.30)

encoded_df["AdjustedPrediction"] = (encoded_df["ChurnProbability"] >= prob_threshold).astype(int)

tab1, tab2, tab3 = st.tabs(["Churn Summary & Key Drivers","Customer Demographics","Customer Lookup"])

with tab1:
    st.title("📊 Customer Churn Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", len(encoded_df))

    with col2:
        st.metric("Churn Rate", f"{encoded_df['ChurnStatus'].mean():.1%}")

    with col3:
        st.metric("Avg Total Spend", f"${merged_df['total_spend'].mean():.2f}")

    st.subheader("Predicted Churn Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(encoded_df["ChurnProbability"], bins=20)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    merged_df["AgeGroup"] = pd.cut(
    merged_df["Age"], bins=[0,30,45,60,100], labels=["<30","30–45","45–60","60+"]
    )
    st.subheader("Churn Rate by Age")
    fig, ax = plt.subplots()
    sns.barplot(data=merged_df, x="AgeGroup", y="ChurnStatus", ax=ax)
    st.pyplot(fig)

    # -------------------------
    # KEY DRIVERS EXPANDER
    # -------------------------
    with st.expander("Key Drivers of Churn (Feature Importance)", expanded=False):
        importances = model.feature_importances_
        feature_df = (
            pd.DataFrame({"Feature": x.columns, "Importance": importances})
            .sort_values("Importance", ascending=False)
        )
        st.bar_chart(feature_df.set_index("Feature"))


with tab2:
    st.title("Customer Demographic Breakdown")

    st.subheader("Churn by Income Level")
    fig, ax = plt.subplots()
    sns.barplot(data=merged_df, x="IncomeLevel", y="ChurnStatus", ax=ax)
    st.pyplot(fig)

    st.subheader("Login Frequency by Churn Status")
    fig, ax = plt.subplots()
    sns.boxplot(data=merged_df, x="Churn Status", y="LoginFrequency", ax=ax)
    st.pyplot(fig)

    st.subheader("Total Spend Distribution")
    fig, ax = plt.subplots()
    sns.histplot(merged_df["total_spend"], bins=20, ax=ax)
    st.pyplot(fig)

with tab3:
    st.title("🔍 Customer Lookup")

    customer_ids = merged_df["CustomerID"].unique()
    selected = st.selectbox("Choose a Customer ID:", customer_ids)

    person = merged_df[merged_df["CustomerID"] == selected].iloc[0]

    st.write("### Customer Snapshot")
    st.write({
        "Age": person["Age"],
        "Income Level": person["IncomeLevel"],
        "Login Frequency": float(person["LoginFrequency"]),
        "Total Spend": person["total_spend"],
        "Churn Probability": round(person["ChurnProbability"], 3),
        "Risk Level": "High Risk" if person["ChurnProbability"] > prob_threshold else "Low Risk"
    })








