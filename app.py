# ==========================================================
# EMOTION ANALYTICS FROM JOURNAL APPS
# FINAL STREAMLIT DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Emotion Analytics for Mental Health",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------------------------
# LOAD MODEL FILES
# ----------------------------------------------------------
@st.cache_resource
def load_assets():
    svm_model = joblib.load("svm_sentiment_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return svm_model, vectorizer, label_encoder

svm_model, vectorizer, label_encoder = load_assets()

# ----------------------------------------------------------
# TEXT PREPROCESSING
# ----------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ----------------------------------------------------------
# EMOTION LEXICON (RULE-BASED)
# ----------------------------------------------------------
emotion_keywords = {
    "Joy": ["happy", "joy", "excited", "grateful", "pleased"],
    "Sadness": ["sad", "lonely", "hopeless", "cry", "down"],
    "Anger": ["angry", "frustrated", "irritated", "mad"],
    "Fear": ["anxious", "worried", "panic", "scared", "stress"],
    "Calm": ["calm", "peace", "relaxed", "content"]
}

def detect_emotions(text):
    scores = {emo: 0 for emo in emotion_keywords}
    for emo, words in emotion_keywords.items():
        for w in words:
            if w in text:
                scores[emo] += 1
    return scores

# ----------------------------------------------------------
# SENTIMENT SCORE
# ----------------------------------------------------------
def sentiment_score(sentiment):
    if sentiment.lower() == "positive":
        return 0.8
    elif sentiment.lower() == "neutral":
        return 0.5
    else:
        return 0.2

# ----------------------------------------------------------
# SESSION HISTORY
# ----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------
st.sidebar.title("📊 Dashboard Menu")
menu = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Live Journal Analysis",
        "Emotion Analytics",
        "Trend & Timeline",
        "Model Evaluation",
        "Insights",
        "History",
        "About"
    ]
)

# ==========================================================
# OVERVIEW
# ==========================================================
if menu == "Overview":
    st.title("🧠 Emotion Analytics for Early Mental Health Detection")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "Linear SVM")
    col2.metric("Feature Extraction", "TF-IDF")
    col3.metric("Analysis Type", "Sentiment + Emotion")
    col4.metric("Platform", "Streamlit")

    st.markdown("""
    This dashboard implements **Emotion Analytics from Journal Apps**
    to support **early mental health detection** using NLP and machine learning.
    """)

# ==========================================================
# LIVE JOURNAL ANALYSIS (FIXED & FINAL)
# ==========================================================
elif menu == "Live Journal Analysis":
    st.title("✍️ Live Journal Entry Analysis")

    text = st.text_area("Enter journal entry:", height=180)

    if st.button("Analyze Entry") and text.strip() != "":
        # Preprocess
        clean = clean_text(text)
        vec = vectorizer.transform([clean])

        # Sentiment prediction
        pred = svm_model.predict(vec)
        sentiment = label_encoder.inverse_transform(pred)[0]

        # Confidence-based neutral handling
        decision = svm_model.decision_function(vec)
        if abs(decision[0]) < 0.25:
            sentiment = "Neutral"

        # Emotion detection
        emotions = detect_emotions(clean)
        if sum(emotions.values()) == 0:
            primary_emotion = "Neutral"
        else:
            primary_emotion = max(emotions, key=emotions.get)

        # Sentiment score
        score = sentiment_score(sentiment)

        # Save history
        st.session_state.history.append({
            "time": datetime.now(),
            "text": text,
            "sentiment": sentiment,
            "emotion": primary_emotion,
            "score": score
        })

        # Display result
        st.success(f"Sentiment: **{sentiment}**")
        st.info(f"Primary Emotion: **{primary_emotion}**")

        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Sentiment Score"},
            gauge={
                "axis": {"range": [0, 1]},
                "steps": [
                    {"range": [0, 0.4], "color": "red"},
                    {"range": [0.4, 0.6], "color": "yellow"},
                    {"range": [0.6, 1], "color": "green"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Emotion bar chart
        fig2, ax = plt.subplots()
        ax.bar(emotions.keys(), emotions.values())
        ax.set_title("Emotion Intensity")
        ax.set_ylabel("Score")
        st.pyplot(fig2)

# ==========================================================
# EMOTION ANALYTICS
# ==========================================================
elif menu == "Emotion Analytics":
    st.title("📊 Emotion Analytics & Heatmap")

    if len(st.session_state.history) == 0:
        st.warning("No data available yet.")
    else:
        df = pd.DataFrame(st.session_state.history)

        # Emotion distribution
        fig, ax = plt.subplots()
        df["emotion"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Emotion Frequency Distribution")
        st.pyplot(fig)

        # Heatmap
        pivot = pd.crosstab(df["emotion"], df["sentiment"])
        fig2, ax2 = plt.subplots()
        sns.heatmap(pivot, annot=True, cmap="coolwarm", ax=ax2)
        ax2.set_title("Emotion vs Sentiment Heatmap")
        st.pyplot(fig2)

# ==========================================================
# TREND & TIMELINE
# ==========================================================
elif menu == "Trend & Timeline":
    st.title("📈 Sentiment Trend Over Time")

    if len(st.session_state.history) == 0:
        st.info("Add journal entries to view trends.")
    else:
        df = pd.DataFrame(st.session_state.history)
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["score"], marker="o")
        ax.set_ylabel("Sentiment Score")
        ax.set_title("Sentiment Timeline")
        st.pyplot(fig)

# ==========================================================
# MODEL EVALUATION
# ==========================================================
elif menu == "Model Evaluation":
    st.title("📉 Model Evaluation & Confusion Matrix")

    # Example visualization (replace with real test data if available)
    y_true = ["Positive", "Negative", "Neutral", "Positive"]
    y_pred = ["Positive", "Neutral", "Neutral", "Positive"]

    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ==========================================================
# INSIGHTS
# ==========================================================
elif menu == "Insights":
    st.title("🧠 Intelligent Insights")

    if len(st.session_state.history) == 0:
        st.info("No insights available.")
    else:
        df = pd.DataFrame(st.session_state.history)
        avg_score = df["score"].mean()

        if avg_score < 0.4:
            st.error("⚠️ Sustained negative emotional trend detected.")
        elif avg_score < 0.6:
            st.warning("⚠️ Emotional state appears unstable.")
        else:
            st.success("✅ Emotional patterns appear stable and positive.")

# ==========================================================
# HISTORY
# ==========================================================
elif menu == "History":
    st.title("📜 Journal Analysis History")
    if len(st.session_state.history) == 0:
        st.info("No history yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history))

# ==========================================================
# ABOUT
# ==========================================================
elif menu == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Project Title:** Emotion Analytics from Journal Apps for Early Mental Health Detection  
    **Model:** Linear SVM  
    **Tech Stack:** NLP, Machine Learning, Streamlit  

    ⚠️ This system is for academic and research purposes only.
    """)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>🧠 Emotion Analytics Dashboard | Final Year Project</center>",
    unsafe_allow_html=True
)

