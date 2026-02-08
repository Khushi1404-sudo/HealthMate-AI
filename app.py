import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# --- 1. SETUP & STYLING (Your CSS) ---
st.set_page_config(page_title="HealthMate AI", layout="centered")

st.markdown(f"""
    <style>
    .stApp {{ background-color: #0a1628; color: white; }}
    .stSlider {{ background-color: #1e293b; padding: 20px; border-radius: 15px; }}
    h1 {{ color: #0ea5e9; text-align: center; }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA PREPROCESSING & AI ---
@st.cache_resource
def build_ai():
    # Preprocessing
    # Replace 'data.csv' with your actual file path
    try:
        df = pd.read_csv('data.csv') 
    except:
        # Dummy data so it works immediately
        data = {'Systolic': np.random.randint(90, 180, 100), 'Diastolic': np.random.randint(60, 110, 100),
                'HeartRate': np.random.randint(50, 150, 100), 'Temp': np.random.uniform(35, 40, 100),
                'Target': np.random.randint(0, 2, 100)}
        df = pd.DataFrame(data)

    df['High_BP'] = ((df['Systolic'] > 140) | (df['Diastolic'] > 90)).astype(int)
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    model = XGBClassifier(n_estimators=50)
    model.fit(X, y)
    return model

final_model = build_ai()

# --- 3. THE WEBSITE UI ---
st.title("🏥 HealthMate AI")
st.write("Personal Health Assistant powered by XGBoost")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🛠 Vitals")
    hr = st.slider("Heart Rate (bpm)", 40, 180, 72)
    temp_f = st.slider("Temperature (°F)", 94.0, 105.0, 98.6)
    sys = st.slider("Systolic BP", 70, 200, 120)
    dia = st.slider("Diastolic BP", 40, 130, 80)

with col2:
    st.subheader("🤖 AI Analysis")
    
    # Logic
    temp_c = (temp_f - 32) * 5/9
    high_bp = 1 if (sys > 140 or dia > 90) else 0
    features = np.array([[sys, dia, hr, temp_c, high_bp]])
    
    prediction = final_model.predict(features)[0]
    is_emergency = (temp_f > 103 or temp_f < 95) or (hr > 130 or hr < 45)

    if is_emergency:
        st.error("### 🛑 EMERGENCY\nSeek medical help immediately.")
    elif prediction == 1:
        st.warning("### ⚠️ WARNING\nHigh-risk pattern detected.")
    else:
        st.success("### ✅ STABLE\nVitals are normal.")