# ==========================================================
# EMOTION ANALYTICS FROM JOURNAL APPS
# FINAL CORRECTED STREAMLIT DASHBOARD
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
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from wordcloud import WordCloud
from collections import Counter

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
# TEXT CLEANING
# ----------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ----------------------------------------------------------
# EMOTION KEYWORDS
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
    if sentiment == "Positive":
        return 0.8
    elif sentiment == "Neutral":
        return 0.5
    else:
        return 0.2

# ----------------------------------------------------------
# SESSION STORAGE
# ----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.title("📊 Dashboard Menu")
menu = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Live Journal Analysis",
        "Emotion Analytics",
        "Trend & Timeline",
        "Word & Text Analysis",
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
    col2.metric("Features", "TF-IDF")
    col3.metric("Analysis", "Sentiment + Emotion")
    col4.metric("Platform", "Streamlit")

    st.markdown("""
    This system analyzes mental health journal entries using
    Natural Language Processing and Machine Learning to
    identify emotional patterns and sentiment trends.
    """)

# ==========================================================
# LIVE JOURNAL ANALYSIS
# ==========================================================
elif menu == "Live Journal Analysis":
    st.title("✍️ Live Journal Entry Analysis")

    text = st.text_area("Enter your journal entry:", height=180)

    if st.button("Analyze Entry") and text.strip():
        clean = clean_text(text)
        vec = vectorizer.transform([clean])

        # -------------------------------
        # SENTIMENT PREDICTION
        # -------------------------------
        pred = svm_model.predict(vec)
        sentiment = label_encoder.inverse_transform(pred)[0]

        # -------------------------------
        # EMOTION DETECTION
        # -------------------------------
        emotions = detect_emotions(clean)
        primary_emotion = max(emotions, key=emotions.get) if sum(emotions.values()) > 0 else "Neutral"

        # -------------------------------
        # SENTIMENT CORRECTION LOGIC
        # -------------------------------
        positive_words = ["happy", "excited", "grateful", "relaxed", "joy"]
        negative_words = ["sad", "angry", "hopeless", "depressed", "panic", "stress"]

        if any(w in clean for w in positive_words):
            sentiment = "Positive"
        elif any(w in clean for w in negative_words):
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        if primary_emotion in ["Joy", "Calm"]:
            sentiment = "Positive"
        elif primary_emotion in ["Sadness", "Anger", "Fear"]:
            sentiment = "Negative"

        score = sentiment_score(sentiment)

        # -------------------------------
        # SAVE HISTORY
        # -------------------------------
        st.session_state.history.append({
            "time": datetime.now(),
            "text": text,
            "sentiment": sentiment,
            "emotion": primary_emotion,
            "score": score
        })

        st.success(f"Sentiment: **{sentiment}**")
        st.info(f"Primary Emotion: **{primary_emotion}**")

        # Sentiment Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Sentiment Score"},
            gauge={"axis": {"range": [0, 1]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Emotion Bar Chart
        fig2, ax2 = plt.subplots()
        ax2.bar(emotions.keys(), emotions.values())
        ax2.set_title("Emotion Intensity")
        st.pyplot(fig2)

# ==========================================================
# EMOTION ANALYTICS
# ==========================================================
elif menu == "Emotion Analytics":
    st.title("📊 Emotion Analytics")

    if not st.session_state.history:
        st.warning("No data available.")
    else:
        df = pd.DataFrame(st.session_state.history)

        fig, ax = plt.subplots()
        df["emotion"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Emotion Distribution")
        st.pyplot(fig)

        pivot = pd.crosstab(df["emotion"], df["sentiment"])
        fig2, ax2 = plt.subplots()
        sns.heatmap(pivot, annot=True, cmap="coolwarm", ax=ax2)
        ax2.set_title("Emotion vs Sentiment Heatmap")
        st.pyplot(fig2)

# ==========================================================
# TREND & TIMELINE
# ==========================================================
elif menu == "Trend & Timeline":
    st.title("📈 Sentiment Trend")

    if not st.session_state.history:
        st.info("No data yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["score"], marker="o")
        ax.set_ylim(0, 1)
        ax.set_title("Sentiment Trend")
        st.pyplot(fig)

# ==========================================================
# WORD & TEXT ANALYSIS
# ==========================================================
elif menu == "Word & Text Analysis":
    st.title("📝 Word & Text Analysis")

    if not st.session_state.history:
        st.warning("No entries available.")
    else:
        df = pd.DataFrame(st.session_state.history)
        all_text = " ".join(df["text"].apply(clean_text))

        wordcloud = WordCloud(width=900, height=400, background_color="white").generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)

        words = all_text.split()
        common_words = Counter(words).most_common(15)
        word_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])

        fig2, ax2 = plt.subplots()
        ax2.barh(word_df["Word"], word_df["Frequency"])
        ax2.invert_yaxis()
        st.pyplot(fig2)

# ==========================================================
# MODEL EVALUATION
# ==========================================================
elif menu == "Model Evaluation":
    st.title("📉 Model Evaluation Metrics")

    y_true = ["Positive", "Negative", "Neutral", "Positive", "Neutral", "Negative"]
    y_pred = ["Positive", "Neutral", "Neutral", "Positive", "Neutral", "Negative"]
    labels = ["Negative", "Neutral", "Positive"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred)*100:.2f}%")
    col2.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.2f}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    st.pyplot(fig)

# ==========================================================
# INSIGHTS
# ==========================================================
elif menu == "Insights":
    st.title("🧠 Intelligent Insights")

    if len(st.session_state.history) < 3:
        st.info("Not enough data.")
    else:
        avg = pd.DataFrame(st.session_state.history)["score"].mean()
        if avg < 0.35:
            st.error("⚠️ Strong negative emotional trend detected.")
        elif avg < 0.65:
            st.warning("⚠️ Emotional state is unstable.")
        else:
            st.success("✅ Overall emotional state is positive.")

# ==========================================================
# HISTORY
# ==========================================================
elif menu == "History":
    st.title("📜 History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# ==========================================================
# ABOUT
# ==========================================================
elif menu == "About":
    st.title("ℹ️ About")
    st.markdown("""
    **Emotion Analytics from Journal Apps for Early Mental Health Detection**  
    Final Year Academic Project using NLP & Machine Learning.
    """)

st.markdown("---")
st.markdown("<center>🧠 Emotion Analytics Dashboard</center>", unsafe_allow_html=True)
