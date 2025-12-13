# ==========================================================
# EMOTION ANALYTICS FROM JOURNAL APPS
# COMPLETE FINAL STREAMLIT DASHBOARD
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
# SENTIMENT SCORE NORMALIZATION
# ----------------------------------------------------------
def sentiment_score(sentiment):
    if sentiment.lower() == "positive":
        return 0.8
    elif sentiment.lower() == "neutral":
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

    if st.button("Analyze Entry") and text.strip() != "":
        clean = clean_text(text)
        vec = vectorizer.transform([clean])

        # Sentiment Prediction
        pred = svm_model.predict(vec)
        sentiment = label_encoder.inverse_transform(pred)[0]

        # Multi-class confidence handling
        decision_scores = svm_model.decision_function(vec)[0]
        sorted_scores = np.sort(decision_scores)
        confidence_gap = sorted_scores[-1] - sorted_scores[-2]

        if confidence_gap < 0.3:
            sentiment = "Neutral"

        # Emotion Detection
        emotions = detect_emotions(clean)
        primary_emotion = (
            max(emotions, key=emotions.get)
            if sum(emotions.values()) > 0 else "Neutral"
        )

        score = sentiment_score(sentiment)

        # Save history
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
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

# ==========================================================
# EMOTION ANALYTICS
# ==========================================================
elif menu == "Emotion Analytics":
    st.title("📊 Emotion Analytics")

    if len(st.session_state.history) == 0:
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
    st.title("📈 Sentiment Trend Over Time")

    if len(st.session_state.history) == 0:
        st.info("No data yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["score"], marker="o")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Sentiment Score")
        ax.set_xlabel("Time")
        ax.set_title("Sentiment Trend (0–1 Scale)")
        st.pyplot(fig)

# ==========================================================
# WORD & TEXT ANALYSIS
# ==========================================================
elif menu == "Word & Text Analysis":
    st.title("📝 Word & Text Analysis")

    if len(st.session_state.history) == 0:
        st.warning("No journal entries available.")
    else:
        df = pd.DataFrame(st.session_state.history)
        all_text = " ".join(df["text"].apply(clean_text))

        # Word Cloud
        wordcloud = WordCloud(
            width=900,
            height=400,
            background_color="white"
        ).generate(all_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Top Words
        words = all_text.split()
        common_words = Counter(words).most_common(15)
        word_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])

        fig2, ax2 = plt.subplots()
        ax2.barh(word_df["Word"], word_df["Frequency"])
        ax2.invert_yaxis()
        ax2.set_title("Top Emotional Words")
        st.pyplot(fig2)

# ==========================================================
# MODEL EVALUATION
# ==========================================================
elif menu == "Model Evaluation":
    st.title("📉 Model Evaluation")

    y_true = ["Positive", "Negative", "Neutral", "Positive", "Neutral"]
    y_pred = ["Positive", "Neutral", "Neutral", "Positive","Negative"]

    labels = ["Negative", "Neutral", "Positive"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Sample Evaluation)")
    st.pyplot(fig)
    
    st.info(
        "Note: This confusion matrix is shown for demonstration purposes, "
        "as live user input does not have ground-truth labels."
    )

# ==========================================================
# INSIGHTS
# ==========================================================
elif menu == "Insights":
    st.title("🧠 Intelligent Insights")

    if len(st.session_state.history) == 0:
        st.info("No insights yet.")
    else:
        avg = pd.DataFrame(st.session_state.history)["score"].mean()
        if avg < 0.4:
            st.error("⚠️ Strong negative emotional trend detected.")
        elif avg < 0.6:
            st.warning("⚠️ Emotional state is unstable.")
        else:
            st.success("✅ Overall emotional state is positive.")

# ==========================================================
# HISTORY
# ==========================================================
elif menu == "History":
    st.title("📜 Analysis History")
    if len(st.session_state.history) == 0:
        st.info("No history yet.")
    else:
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
st.markdown(
    "<center>🧠 Emotion Analytics Dashboard | Final Year Project</center>",
    unsafe_allow_html=True
)

