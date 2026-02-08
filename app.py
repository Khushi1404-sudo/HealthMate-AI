import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components

# --- 1. DATA PREPROCESSING (From your Colab) ---
@st.cache_resource
def train_model():
    # Load your data
    df = pd.read_csv('data_clean.csv')
    
    # Feature Engineering
    df['High_BP'] = ((df['Systolic'] > 140) | (df['Diastolic'] > 90)).astype(int)
    
    # Split & Train
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    weight_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = XGBClassifier(scale_pos_weight=weight_ratio, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

model = train_model()

# --- 2. THE UI (Your Exact Design) ---
st.set_page_config(layout="wide")

# This reads your HTML file and shows it on the screen
with open("index.html", "r") as f:
    html_content = f.read()

# This displays your design exactly as you wrote it
components.html(html_content, height=900, scrolling=True)
