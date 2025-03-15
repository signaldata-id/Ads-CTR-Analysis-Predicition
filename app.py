import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ========== STREAMLIT UI ========== #
st.set_page_config(page_title="Ads CTR Prediction - Signal Data Education", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    div.stRadio > label {
        font-size: 18px !important;
        font-weight: bold !important;
        color: #007BFF !important;
    }
    div.stRadio > div {
        display: flex;
        justify-content: center;
    }
    .centered-text {
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# **Tampilan Navigasi di Tengah Halaman**
st.markdown("<h1 style='text-align: center;'>📊 Ads CTR Prediction App</h1>", unsafe_allow_html=True)
page = st.radio(
    "Pilih Halaman:",
    ["CTR Prediction", "Looker Dashboard", "Team Member"],
    horizontal=True
)

if page == "CTR Prediction":
    # Load model dan preprocessing tools
    xgb_model = joblib.load("xgboost_ctr_model.pkl")
    label_encoder = joblib.load("label_encoders.pkl")
    label_encoder_target = joblib.load("label_encoder_target.pkl")

    # Definisi ulang fitur kategorikal dan numerik
    feature_columns = ['Age', 'Gender', 'Location', 'Platform', 
                    'Ad_Type', 'Product_Type', 'Sentiment_Score', 'Target_Audience']

    categorical_features = ["Gender", "Location", "Platform", 
                            "Ad_Type", "Product_Type", "Target_Audience"]

    # ========== HEADER ========== #
    st.markdown("<h1 style='text-align: center;'>📊 Ads CTR Prediction by SignalData</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #667;'>Predict Click-Through Rate (CTR) for Your Ads Campaign</h3>", unsafe_allow_html=True)

    # ========== INPUT FORM ========== #
    st.sidebar.header("📝 Input Your Data")

    age = st.sidebar.number_input("Age", min_value=18, max_value=59, value=25)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    location = st.sidebar.selectbox("Location", ['Surabaya', 'Medan', 'Bandung', 'Jakarta', 'Makassar'])
    platform = st.sidebar.selectbox("Platform", ["Meta Ads", "TikTok"])
    ad_type = st.sidebar.selectbox("Ad Type", ['Carousel', 'Video', 'Image'])
    product_type = st.sidebar.selectbox("Product Type", ['Learning Plan', 'Live Code', 'Bootcamp'])
    sentiment_score = st.sidebar.slider("Sentiment Score", 0.0, 1.0, 0.5)
    target_audience = st.sidebar.selectbox("Target Audience", ["Beginner", "Intermediate", "Advanced"])

    # ========== PENGOLAHAN DATA ========== #
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Location": [location],
        "Platform": [platform],
        "Ad_Type": [ad_type],
        "Product_Type": [product_type],
        "Sentiment_Score": [sentiment_score],
        "Target_Audience": [target_audience]
    })

    # Encoding variabel kategorikal
    for col in categorical_features:
        if col in input_df and col in label_encoder:
            le = label_encoder.get(col)
            if le:
                input_df[col] = le.transform(input_df[col])

    # Pastikan urutan kolom sesuai model
    input_df = input_df[feature_columns]

    # ========== PREDIKSI CTR ========== #
    if st.sidebar.button("🔍 Predict CTR"):
        y_pred_encoded = xgb_model.predict(input_df)
        y_pred_proba = xgb_model.predict_proba(input_df)
        
        # Konversi hasil prediksi ke label asli
        y_pred_label = label_encoder_target.inverse_transform(y_pred_encoded)
        confidence_score = np.max(y_pred_proba, axis=1)[0]
        
        # Tampilkan hasil prediksi
        st.markdown("<h3 style='text-align: center;'>📢 Predicted CTR Category:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #007BFF;'>{y_pred_label[0]}</div>", unsafe_allow_html=True)
        
        # Tampilkan confidence score
        st.markdown("<h3 style='text-align: center;'>🎯 Confidence Score:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #28a745;'>{round(confidence_score * 100, 2)}%</div>", unsafe_allow_html=True)

elif page == "Looker Dashboard":
    st.markdown("<h1 style='text-align: center;'>📊 Ads CTR Dashboard</h1>", unsafe_allow_html=True)

    looker_url = "https://lookerstudio.google.com/embed/reporting/f522afb9-7fbc-4aa5-8927-2bd6d4e62ccd/page/p_2zfsayu9pd"

    st.components.v1.iframe(looker_url, width=1200, height=800)

elif page == "Team Member":
    # ========== TIM SIGNAL DATA EDUCATION ========== #
    st.markdown("---")
    with st.expander("👥 View Team Members"):
        st.markdown("### 👥 Signal Data Education Team")
        
        st.markdown("#### 📌 Project Management")
        st.markdown("- **Abdiel** - [LinkedIn](#)")

        st.markdown("#### 🤝 Project Support")
        st.markdown("- Elisabeth Sengkey - [LinkedIn](#)")
        st.markdown("- Erlita - [LinkedIn](#)")

        st.markdown("#### 🛠 Data Engineering Team")
        st.markdown("- Renona - [LinkedIn](#)")
        st.markdown("- Agung Bakti - [LinkedIn](#)")

        st.markdown("#### 📊 Data Analytics Team")
        st.markdown("- Arella - [LinkedIn](#)")
        st.markdown("- Katerina - [LinkedIn](#)")
        st.markdown("- Nofrani - [LinkedIn](#)")

        st.markdown("#### 🤖 Machine Learning Team")
        st.markdown("- Roma Mantiri - [LinkedIn](#)")
        st.markdown("- Bagas Akbar Maulana - [LinkedIn](#)")
