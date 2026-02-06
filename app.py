# ==========================================================
# EMOTION ANALYTICS FROM JOURNAL APPS
# FINAL STREAMLIT DASHBOARD (FULLY FIXED)
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
from wordcloud import WordCloud
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split

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
# WORD LISTS (100+ EACH)
# ----------------------------------------------------------
positive_words = [
    "happy","joy","joyful","cheerful","delighted","pleased","content","satisfied",
    "good","great","excellent","amazing","awesome","fantastic","wonderful","nice",
    "well","better","best","positive","optimistic","hopeful","grateful","thankful",
    "blessed","calm","peaceful","relaxed","comfortable","safe","secure","confident",
    "proud","successful","winning","motivated","energetic","fresh","strong","healthy",
    "fine","okay","stable","balanced","focused","clear","smiling","laughing",
    "enjoying","love","loved","lovely","kind","support","excited","inspired",
    "productive","progress","growth","achievement","relief","fun","bright"
]

negative_words = [
    "sad","unhappy","depressed","depression","down","low","lonely","alone",
    "hopeless","helpless","worthless","tired","exhausted","burnout","weak",
    "sick","pain","hurt","cry","crying","angry","mad","furious","annoyed",
    "frustrated","stress","stressed","anxious","anxiety","worried","fear",
    "panic","scared","nervous","upset","confused","lost","empty","broken",
    "failure","failed","hate","guilty","shame","regret","pressure",
    "burden","overwhelmed","bad","worse","worst","miserable","dark"
]

neutral_words = [
    "fine","okay","normal","average","usual","routine","regular","same",
    "unchanged","stable","balanced","neutral","simple","moderate","calm",
    "quiet","waiting","thinking","working","studying","learning","reading",
    "writing","walking","sitting","planning","checking","reviewing",
    "maybe","perhaps","unsure","today","morning","evening","time",
    "currently","present","state","condition","situation","steady","alright"
]

# ----------------------------------------------------------
# RULE BASED SENTIMENT
# ----------------------------------------------------------
def rule_based_sentiment(text):
    pos = sum(w in text for w in positive_words)
    neg = sum(w in text for w in negative_words)
    neu = sum(w in text for w in neutral_words)

    if pos > neg and pos > neu:
        return "Positive"
    elif neg > pos and neg > neu:
        return "Negative"
    else:
        return "Neutral"

def sentiment_score(sentiment):
    return {"Positive": 0.8, "Neutral": 0.5, "Negative": 0.2}[sentiment]

# ----------------------------------------------------------
# SESSION STATE
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
    col1.metric("Algorithm", "SVM")
    col2.metric("Features", "TF-IDF")
    col3.metric("Evaluation", "Accuracy / F1")
    col4.metric("Platform", "Streamlit")

# ==========================================================
# LIVE JOURNAL
# ==========================================================
elif menu == "Live Journal Analysis":
    st.title("✍️ Live Journal Entry Analysis")

    text = st.text_area("Enter your journal entry", height=180)

    if st.button("Analyze") and text.strip():
        clean = clean_text(text)
        sentiment = rule_based_sentiment(clean)

        if sentiment == "Neutral":
            vec = vectorizer.transform([clean])
            pred = svm_model.predict(vec)
            sentiment = label_encoder.inverse_transform(pred)[0]

        score = sentiment_score(sentiment)

        st.session_state.history.append({
            "time": datetime.now(),
            "text": text,
            "sentiment": sentiment,
            "score": score
        })

        st.success(f"Sentiment: **{sentiment}**")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Sentiment Score"},
            gauge={"axis": {"range": [0, 1]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# EMOTION ANALYTICS
# ==========================================================
elif menu == "Emotion Analytics":
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.bar_chart(df["sentiment"].value_counts())

# ==========================================================
# TREND
# ==========================================================
elif menu == "Trend & Timeline":
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        plt.plot(df["time"], df["score"], marker="o")
        plt.ylim(0, 1)
        st.pyplot(plt.gcf())

# ==========================================================
# WORD CLOUD
# ==========================================================
elif menu == "Word & Text Analysis":
    if st.session_state.history:
        text_all = " ".join(pd.DataFrame(st.session_state.history)["text"])
        wc = WordCloud(width=800, height=400).generate(text_all)
        plt.imshow(wc)
        plt.axis("off")
        st.pyplot(plt.gcf())

# ==========================================================
# MODEL EVALUATION (FIXED)
# ==========================================================
elif menu == "Model Evaluation":
    st.title("📊 Model Evaluation")

    file = st.file_uploader("Upload CSV (text,label)", type="csv")

    if file:
        df = pd.read_csv(file)

        X = df["text"].apply(clean_text)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_test_vec = vectorizer.transform(X_test)
        y_pred = svm_model.predict(X_test_vec)
        y_pred = label_encoder.inverse_transform(y_pred)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2f}")
        col4.metric("F1", f"{f1_score(y_test, y_pred, average='weighted'):.2f}")

        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        st.pyplot(plt.gcf())

# ==========================================================
# INSIGHTS
# ==========================================================
elif menu == "Insights":
    if len(st.session_state.history) >= 3:
        avg = pd.DataFrame(st.session_state.history)["score"].mean()
        if avg < 0.35:
            st.error("⚠️ Strong negative trend detected")
        elif avg < 0.65:
            st.warning("⚠️ Emotional instability detected")
        else:
            st.success("✅ Positive trend")

# ==========================================================
# HISTORY
# ==========================================================
elif menu == "History":
    st.dataframe(pd.DataFrame(st.session_state.history))

# ==========================================================
# ABOUT
# ==========================================================
elif menu == "About":
    st.markdown("Final Year Project – Emotion Analytics using NLP & ML")

st.markdown("---")
st.markdown("<center>🧠 Emotion Analytics Dashboard</center>", unsafe_allow_html=True)
