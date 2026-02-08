import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import json

# --- 1. DATA PREPROCESSING (With Auto-Cleaning) ---
@st.cache_resource
def train_model():
    df = pd.read_csv('Patient_Dataset.csv')
    
    # Force column names to be clean: lowercase and no spaces
    df.columns = df.columns.str.strip().str.lower()
    
    # Match the logic to your new clean names
    # Assuming your CSV has columns that include 'systolic', 'diastolic', and 'target'
    df['high_bp'] = ((df['systolic'] > 140) | (df['diastolic'] > 90)).astype(int)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    weight_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = XGBClassifier(scale_pos_weight=weight_ratio, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model, X.columns.tolist()

# Load Model
model, feature_names = train_model()

# --- 2. THE UI ---
st.set_page_config(layout="wide")

with open("index.html", "r") as f:
    html_code = f.read()

# This receives the "Message" from the Javascript
val = components.html(html_code, height=700)

if val:
    # Decode the data sent from the HTML sliders
    user_input = json.loads(val)
    hr = float(user_input['hr'])
    temp = float(user_input['temp'])
    sys = float(user_input['sys'])
    dia = float(user_input['dia'])
    high_bp = 1 if (sys > 140 or dia > 90) else 0

    # Prepare data for AI (Must match the order of your CSV columns)
    # We create a dictionary to ensure the names match exactly what the model saw
    input_df = pd.DataFrame([{
        'systolic': sys,
        'diastolic': dia,
        'heartrate': hr, # Make sure these names match your CSV columns!
        'temp': temp,
        'high_bp': high_bp
    }])
    
    # If your CSV has more columns, you may need to add them here.
    # For now, let's predict:
    prediction = model.predict(input_df)[0]
    
    # Send result BACK to the HTML design
    title = "⚠️ HIGH RISK" if prediction == 1 else "✅ STABLE"
    msg = "The AI detected patterns associated with health risks." if prediction == 1 else "Your vitals are within the normal AI range."
    color = "#ef4444" if prediction == 1 else "#10b981"
    
    # Update the HTML component with the result
    components.html(html_code.replace("", ""), height=700)
    st.sidebar.metric("AI Status", title)
    st.sidebar.write(msg)
