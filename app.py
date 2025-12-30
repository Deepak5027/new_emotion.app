import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Emotion Analytics Dashboard", layout="wide")

# ---------------- LOAD MODELS ----------------
svm_model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ---------------- UTILS ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = svm_model.predict(vec)
    return label_encoder.inverse_transform(pred)[0]

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "Live Journal Analysis",
        "Trends & Timeline",
        "Word & Text Analysis",
        "Insights",
        "Model Evaluation",
        "History",
        "About"
    ]
)

# ---------------- LIVE JOURNAL ----------------
if menu == "Live Journal Analysis":
    st.title("📝 Live Journal Analysis")

    text = st.text_area("Write your journal entry")

    if st.button("Analyze"):
        if text.strip():
            sentiment = predict_sentiment(text)
            st.subheader(f"Predicted Emotion: **{sentiment}**")

            st.session_state.history.append({
                "time": datetime.now(),
                "text": text,
                "sentiment": sentiment
            })

# ---------------- TRENDS & TIMELINE ----------------
elif menu == "Trends & Timeline":
    st.title("📈 Emotion Trends Over Time")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        timeline = df.groupby(df["time"].dt.date)["sentiment"].value_counts().unstack().fillna(0)

        st.line_chart(timeline)
    else:
        st.info("No data available yet.")

# ---------------- WORD ANALYSIS ----------------
elif menu == "Word & Text Analysis":
    st.title("🔤 Word Frequency Analysis")

    if st.session_state.history:
        all_text = " ".join([clean_text(i["text"]) for i in st.session_state.history])
        words = all_text.split()
        word_freq = Counter(words).most_common(20)

        df_words = pd.DataFrame(word_freq, columns=["Word", "Count"])
        st.bar_chart(df_words.set_index("Word"))
    else:
        st.info("No text data available.")

# ---------------- INSIGHTS ----------------
elif menu == "Insights":
    st.title("🧠 Insights Dashboard")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        counts = df["sentiment"].value_counts()

        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", counts.get("Positive", 0))
        col2.metric("Negative", counts.get("Negative", 0))
        col3.metric("Neutral", counts.get("Neutral", 0))
    else:
        st.info("No insights yet.")

# ---------------- MODEL EVALUATION ----------------
elif menu == "Model Evaluation":
    st.title("📊 Model Evaluation")

    file = st.file_uploader("Upload CSV (text, sentiment)", type=["csv"])

    if file:
        df = pd.read_csv(file)

        X = vectorizer.transform(df["text"].astype(str).apply(clean_text))
        y_true = label_encoder.transform(df["sentiment"])

        y_pred = svm_model.predict(X)

        st.subheader("Accuracy")
        st.write(accuracy_score(y_true, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# ---------------- HISTORY ----------------
elif menu == "History":
    st.title("📜 Analysis History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.info("No history available.")

# ---------------- ABOUT ----------------
elif menu == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **Emotion Analytics Dashboard**

    - Live Journal Emotion Detection
    - Machine Learning (SVM)
    - Real-time Insights & Trends
    - CSV Model Evaluation
    - Final Year Project Ready
    """)

