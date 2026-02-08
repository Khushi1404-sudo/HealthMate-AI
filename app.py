import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import json

# --- 1. DATA PREPROCESSING (Matched to your CSV) ---
@st.cache_resource
def train_ai():
    df = pd.read_csv('Patient_Dataset.csv')
    
    # Split "116/84" into 116 and 84
    bp = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
    df['systolic'] = pd.to_numeric(bp[0])
    df['diastolic'] = pd.to_numeric(bp[1])
    df['hr'] = df['Heart Rate (bpm)']
    df['temp_c'] = df['Temperature (°C)']
    df['high_bp'] = ((df['systolic'] > 140) | (df['diastolic'] > 90)).astype(int)
    
    X = df[['hr', 'temp_c', 'systolic', 'diastolic', 'high_bp']]
    y = df['Target']
    
    model = XGBClassifier().fit(X, y)
    return model

model = train_ai()

# --- 2. THE UI LOGIC ---
st.set_page_config(layout="centered")

# Default values
params = {"HR": "72", "TEMP": "98.6", "SYS": "120", "DIA": "80", 
          "DISPLAY": "none", "TITLE": "", "MSG": "", "COLOR": "transparent"}

with open("index.html", "r") as f:
    template = f.read()

# When the user clicks "Analyze"
val = components.html(template.replace("{{DISPLAY}}", "none"), height=600)

if val:
    data = json.loads(val)
    hr, temp_f, sys, dia = float(data['hr']), float(data['temp']), float(data['sys']), float(data['dia'])
    
    # AI Prediction
    temp_c = (temp_f - 32) * 5/9
    high_bp = 1 if (sys > 140 or dia > 90) else 0
    pred = model.predict(np.array([[hr, temp_c, sys, dia, high_bp]]))[0]
    
    # Update Design with Result
    params.update({"HR": str(hr), "TEMP": str(temp_f), "SYS": str(sys), "DIA": str(dia), "DISPLAY": "block"})
    if pred == 1:
        params.update({"TITLE": "🛑 HIGH RISK", "MSG": "AI detects a high-risk health pattern.", "COLOR": "#ef4444"})
    else:
        params.update({"TITLE": "✅ STABLE", "MSG": "Vitals are within normal range.", "COLOR": "#10b981"})
    
    # Re-render with result
    final_html = template
    for key, value in params.items():
        final_html = final_html.replace(f"{{{{{key}}}}}", value)
    components.html(final_html, height=750)
