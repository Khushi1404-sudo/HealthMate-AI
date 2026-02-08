import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import json

# --- 1. DATA PREPROCESSING ---
@st.cache_resource
def train_model():
    df = pd.read_csv('data_clean.csv')
    df['High_BP'] = ((df['Systolic'] > 140) | (df['Diastolic'] > 90)).astype(int)
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    weight_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = XGBClassifier(scale_pos_weight=weight_ratio, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

final_model = train_model()

# --- 2. THE BRIDGE ---
st.set_page_config(layout="wide", page_title="HealthMate AI")

# This is the magic part: it receives the slider values from your HTML
with open("index.html", "r") as f:
    html_content = f.read()

# We "call" the component and it returns data if the user clicked 'Analyze'
user_data = components.html(html_content, height=800)

# If user_data is sent from JS, we run the AI!
if user_data:
    data = json.loads(user_data)
    hr, temp, sys, dia = data['hr'], data['temp'], data['sys'], data['dia']
    
    # Preprocess input
    high_bp = 1 if (float(sys) > 140 or float(dia) > 90) else 0
    features = np.array([[float(hr), float(temp), float(sys), float(dia), high_bp]])
    
    # Predict
    prediction = final_model.predict(features)[0]
    
    if prediction == 1:
        st.warning("⚠️ AI Prediction: High Risk detected based on your vitals.")
    else:
        st.success("✅ AI Prediction: Your vitals appear stable.")
