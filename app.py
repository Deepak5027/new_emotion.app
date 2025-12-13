# ==========================================================
# ADVANCED MENTAL HEALTH ANALYTICS DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Analytics Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------
@st.cache_resource
def load_assets():
    svm_model = joblib.load("svm_sentiment_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return svm_model, vectorizer, label_encoder

svm_model, vectorizer, label_encoder = load_assets()

# ----------------------------------------------------------
# TEXT CLEANING
# ----------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ----------------------------------------------------------
# EMOTION DETECTION (RULE-BASED – EXAM SAFE)
# ----------------------------------------------------------
emotion_keywords = {
    "Joy": ["happy", "joy", "excited", "grateful", "positive"],
    "Sadness": ["sad", "down", "hopeless", "cry", "lonely"],
    "Anger": ["angry", "frustrated", "irritated", "mad"],
    "Fear": ["anxious", "scared", "fear", "worried", "panic"],
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
# SENTIMENT SCORE (CONTINUOUS)
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
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Live Analysis", "Emotion Analytics", "History", "Insights", "About"]
)

# ==========================================================
# OVERVIEW
# ==========================================================
if page == "Overview":
    st.title("🧠 Mental Health Emotion & Sentiment Analytics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "Linear SVM")
    col2.metric("NLP", "TF-IDF")
    col3.metric("Task", "Sentiment + Emotion")
    col4.metric("Deployment", "Streamlit")

    st.markdown("""
    This intelligent dashboard analyzes **mental health journal entries**
    to extract **sentiment, emotions, behavioral patterns, and insights**.
    """)

# ==========================================================
# LIVE ANALYSIS
# ==========================================================
elif page == "Live Analysis":
    st.title("✍️ Live Journal Analysis")

    text = st.text_area("Enter your journal entry:", height=180)

    if st.button("Analyze"):
        clean = clean_text(text)
        vec = vectorizer.transform([clean])
        pred = svm_model.predict(vec)
        sentiment = label_encoder.inverse_transform(pred)[0]

        score = sentiment_score(sentiment)
        emotions = detect_emotions(clean)
        primary_emotion = max(emotions, key=emotions.get)

        # Save history
        st.session_state.history.append({
            "time": datetime.now(),
            "text": text,
            "sentiment": sentiment,
            "emotion": primary_emotion,
            "score": score
        })

        # ---------------------------
        # SENTIMENT RESULT
        # ---------------------------
        st.success(f"Sentiment: **{sentiment.upper()}**")
        st.info(f"Primary Emotion: **{primary_emotion}**")

        # ---------------------------
        # SENTIMENT SCORE GAUGE
        # ---------------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Sentiment Score"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 0.4], "color": "red"},
                    {"range": [0.4, 0.6], "color": "yellow"},
                    {"range": [0.6, 1], "color": "green"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------
        # EMOTION BAR CHART
        # ---------------------------
        fig2, ax = plt.subplots()
        ax.bar(emotions.keys(), emotions.values())
        ax.set_title("Emotion Intensity")
        ax.set_ylabel("Score")
        st.pyplot(fig2)

# ==========================================================
# EMOTION ANALYTICS
# ==========================================================
elif page == "Emotion Analytics":
    st.title("📈 Emotion Analytics")

    if len(st.session_state.history) == 0:
        st.warning("No data available yet.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        # Emotion distribution
        fig, ax = plt.subplots()
        df_hist["emotion"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Emotion Distribution")
        st.pyplot(fig)

        # Sentiment over time
        fig2, ax2 = plt.subplots()
        ax2.plot(df_hist["time"], df_hist["score"], marker="o")
        ax2.set_title("Sentiment Trend Over Time")
        ax2.set_ylabel("Sentiment Score")
        st.pyplot(fig2)

# ==========================================================
# HISTORY
# ==========================================================
elif page == "History":
    st.title("📜 Analysis History")

    if len(st.session_state.history) == 0:
        st.info("No history yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df[["time", "sentiment", "emotion", "score"]])

# ==========================================================
# GPT-STYLE INSIGHTS (RULE-BASED)
# ==========================================================
elif page == "Insights":
    st.title("🧠 Intelligent Insights")

    if len(st.session_state.history) == 0:
        st.info("Add some entries to get insights.")
    else:
        df = pd.DataFrame(st.session_state.history)
        avg_score = df["score"].mean()

        if avg_score < 0.4:
            st.error("⚠️ Overall emotional tone indicates prolonged negativity.")
        elif avg_score < 0.6:
            st.warning("⚠️ Emotional state appears mixed or unstable.")
        else:
            st.success("✅ Emotional patterns appear stable and positive.")

        st.markdown("""
        **Insight Explanation:**
        - Based on sentiment score trends
        - Emotion frequency distribution
        - No personal data stored permanently
        """)

# ==========================================================
# ABOUT
# ==========================================================
elif page == "About":
    st.title("ℹ️ About")

    st.markdown("""
    **Project:** Mental Health Emotion Analytics  
    **Model:** Linear SVM + NLP  
    **Purpose:** Early emotional pattern detection  

    ⚠️ This system is for educational and research use only.
    """)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown(
    "<center>🧠 Intelligent Mental Health Analytics Dashboard</center>",
    unsafe_allow_html=True
)
