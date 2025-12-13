# ======================================================
# Mental Health Sentiment Analysis Dashboard
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Sentiment Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------------------------------
# LOAD MODEL & FILES
# ------------------------------------------------------
@st.cache_resource
def load_models():
    svm_model = joblib.load("svm_sentiment_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return svm_model, vectorizer, label_encoder

svm_model, vectorizer, label_encoder = load_models()

# ------------------------------------------------------
# TEXT CLEANING FUNCTION
# ------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
st.sidebar.title("📊 Dashboard Controls")
menu = st.sidebar.radio(
    "Navigate",
    ["Overview", "Live Prediction", "Model Performance", "Insights", "About"]
)

# ------------------------------------------------------
# OVERVIEW PAGE
# ------------------------------------------------------
if menu == "Overview":
    st.title("🧠 Mental Health Sentiment Analysis")
    st.markdown("""
    This intelligent dashboard analyzes mental health journal entries using  
    **Natural Language Processing and Machine Learning (Linear SVM)**.

    **Purpose:** Early detection of emotional patterns from written journals.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Used", "Linear SVM")
    col2.metric("Feature Method", "TF-IDF")
    col3.metric("Task", "Sentiment Classification")

# ------------------------------------------------------
# LIVE PREDICTION
# ------------------------------------------------------
elif menu == "Live Prediction":
    st.title("✍️ Live Sentiment Prediction")

    user_text = st.text_area(
        "Enter a mental health journal entry:",
        height=180
    )

    if st.button("Predict Sentiment"):
        cleaned = clean_text(user_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = svm_model.predict(vectorized)
        sentiment = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🧠 Predicted Sentiment: **{sentiment.upper()}**")

# ------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------
elif menu == "Model Performance":
    st.title("📈 Model Performance Evaluation")

    # Dummy accuracy (replace with your actual value if needed)
    final_accuracy = 0.50

    st.metric("Final Model Accuracy", f"{final_accuracy*100:.2f}%")

    # Accuracy Bar
    fig, ax = plt.subplots()
    ax.bar(["Linear SVM"], [final_accuracy])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Final Model Accuracy")
    st.pyplot(fig)

# ------------------------------------------------------
# INSIGHTS PAGE
# ------------------------------------------------------
elif menu == "Insights":
    st.title("🔍 Prediction Behavior & Insights")

    sample_texts = [
        "I feel very anxious and stressed today",
        "I am feeling calm and hopeful",
        "Everything feels overwhelming"
    ]

    predictions = []
    for text in sample_texts:
        clean = clean_text(text)
        vec = vectorizer.transform([clean])
        pred = svm_model.predict(vec)
        predictions.append(label_encoder.inverse_transform(pred)[0])

    insight_df = pd.DataFrame({
        "Journal Text": sample_texts,
        "Predicted Sentiment": predictions
    })

    st.dataframe(insight_df, use_container_width=True)

# ------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------
elif menu == "About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    **Project Name:** Mental Health Journal Sentiment Analysis  
    **Model:** Linear Support Vector Machine (SVM)  
    **Tech Stack:** Python, NLP, Streamlit  

    **Use Case:**  
    - Detect emotional patterns  
    - Support early mental health insights  

    ⚠️ *This system is for educational purposes only.*
    """)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>🧠 Mental Health Analytics | Final Year Project</center>",
    unsafe_allow_html=True
)


