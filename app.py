import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Emotion & Sentiment Analyzer",
    layout="wide"
)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    svm_model = joblib.load("models/svm_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return svm_model, vectorizer, label_encoder

svm_model, vectorizer, label_encoder = load_models()

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------- SENTIMENT PREDICTION --------------------
def predict_sentiment(text):
    text = clean_text(text)
    X = vectorizer.transform([text])

    # SAFE prediction (NO predict_proba)
    decision_scores = svm_model.decision_function(X)

    if len(decision_scores.shape) == 1:
        score = decision_scores[0]
    else:
        score = np.max(decision_scores)

    # Threshold logic
    if score > 0.3:
        return "Positive"
    elif score < -0.3:
        return "Negative"
    else:
        return "Neutral"

# -------------------- SIDEBAR --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Live Journal Analysis", "Model Evaluation"]
)

# =====================================================
# 📝 LIVE JOURNAL ANALYSIS
# =====================================================
if page == "Live Journal Analysis":

    st.title("📝 Live Journal Sentiment Analysis")

    user_text = st.text_area(
        "Write your thoughts below 👇",
        height=200,
        placeholder="I feel happy today because I achieved my goals..."
    )

    if st.button("Analyze Emotion"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            sentiment = predict_sentiment(user_text)

            if sentiment == "Positive":
                st.success("😊 Positive Emotion Detected")
            elif sentiment == "Negative":
                st.error("😞 Negative Emotion Detected")
            else:
                st.info("😐 Neutral Emotion Detected")

# =====================================================
# 📊 MODEL EVALUATION
# =====================================================
if page == "Model Evaluation":

    st.title("📊 Model Evaluation")

    st.write("Upload CSV with columns: **text**, **sentiment**")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if not {"text", "sentiment"}.issubset(df.columns):
            st.error("CSV must contain 'text' and 'sentiment' columns")
        else:
            df["text"] = df["text"].astype(str).apply(clean_text)

            X = vectorizer.transform(df["text"])
            y_true = label_encoder.transform(df["sentiment"])

            # Use predict(), NOT predict_proba()
            y_pred = svm_model.predict(X)

            acc = accuracy_score(y_true, y_pred)
            st.success(f"Accuracy: {acc:.4f}")

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(
                y_true,
                y_pred,
                target_names=label_encoder.classes_,
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)
