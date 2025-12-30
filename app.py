# ==========================================================
# EMOTION ANALYTICS FROM JOURNAL APPS
# FINAL STREAMLIT DASHBOARD (ERROR FIXED)
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
from collections import Counter
from wordcloud import WordCloud

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

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
    text = str(text).lower()
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
    col1.metric("Model", "SVM")
    col2.metric("Features", "TF-IDF")
    col3.metric("Evaluation", "ROC–AUC")
    col4.metric("Platform", "Streamlit")

# ==========================================================
# LIVE JOURNAL ANALYSIS (FIXED)
# ==========================================================
elif menu == "Live Journal Analysis":
    st.title("✍️ Live Journal Entry Analysis")

    text = st.text_area("Enter your journal entry:", height=180)

    if st.button("Analyze Entry") and text.strip():
        clean = clean_text(text)
        vec = vectorizer.transform([clean])

        # ML prediction
        pred_encoded = svm_model.predict(vec)
        sentiment = label_encoder.inverse_transform(pred_encoded)[0]

        # Emotion detection
        emotions = detect_emotions(clean)
        primary_emotion = max(emotions, key=emotions.get) if sum(emotions.values()) > 0 else "Neutral"

        # Smart refinement
        positive_words = ["happy", "excited", "grateful", "relaxed", "joy", "peace"]
        negative_words = ["sad", "angry", "hopeless", "depressed", "panic", "stress"]

        if sentiment == "Neutral":
            if any(w in clean for w in positive_words):
                sentiment = "Positive"
            elif any(w in clean for w in negative_words):
                sentiment = "Negative"

        if sentiment == "Neutral":
            if primary_emotion in ["Joy", "Calm"]:
                sentiment = "Positive"
            elif primary_emotion in ["Sadness", "Anger", "Fear"]:
                sentiment = "Negative"

        score = sentiment_score(sentiment)

        st.session_state.history.append({
            "time": datetime.now(),
            "text": text,
            "sentiment": sentiment,
            "emotion": primary_emotion,
            "score": score
        })

        st.success(f"Sentiment: **{sentiment}**")
        st.info(f"Primary Emotion: **{primary_emotion}**")

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
    st.title("📊 Emotion Analytics")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig, ax = plt.subplots()
        df["emotion"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No data available.")

# ==========================================================
# TREND & TIMELINE
# ==========================================================
elif menu == "Trend & Timeline":
    st.title("📈 Sentiment Trend")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["score"], marker="o")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    else:
        st.info("No data yet.")

# ==========================================================
# WORD & TEXT ANALYSIS
# ==========================================================
elif menu == "Word & Text Analysis":
    st.title("📝 Word & Text Analysis")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        all_text = " ".join(df["text"].apply(clean_text))
        wc = WordCloud(width=900, height=400, background_color="white").generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("No entries available.")

# ==========================================================
# ==========================================================
# MODEL EVALUATION + HYPERPARAMETER TUNING (FIXED)
# ==========================================================
elif menu == "Model Evaluation":
    st.title("📉 Model Evaluation")

    file = st.file_uploader(
        "Upload CSV with columns: text, sentiment",
        type=["csv"]
    )

    if file:
        df = pd.read_csv(file)
        df["clean_text"] = df["text"].apply(clean_text)

        X = vectorizer.transform(df["clean_text"])

        # ✅ Encode true labels
        y_true_encoded = label_encoder.transform(df["sentiment"])
        y_pred_encoded = svm_model.predict(X)

        # ---------------- METRICS ----------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Accuracy",
            f"{accuracy_score(y_true_encoded, y_pred_encoded)*100:.2f}%"
        )
        col2.metric(
            "Precision",
            f"{precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0):.2f}"
        )
        col3.metric(
            "Recall",
            f"{recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0):.2f}"
        )
        col4.metric(
            "F1 Score",
            f"{f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0):.2f}"
        )

        # ---------------- CONFUSION MATRIX ----------------
        labels = label_encoder.classes_
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)

        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels
        )
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        # ---------------- ROC–AUC ----------------
        if hasattr(svm_model, "decision_function"):
            y_scores = svm_model.decision_function(X)
            y_bin = label_binarize(
                y_true_encoded,
                classes=range(len(labels))
            )

            fig_roc, ax_roc = plt.subplots()
            for i, label in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
                ax_roc.plot(fpr, tpr, label=label)

            ax_roc.plot([0, 1], [0, 1], "k--")
            ax_roc.set_title("ROC–AUC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

            st.success(
                f"Weighted ROC–AUC: "
                f"{roc_auc_score(y_bin, y_scores, average='weighted'):.2f}"
            )

        # ==================================================
        # 🔥 HYPERPARAMETER TUNING UI
        # ==================================================
        st.markdown("---")
        st.subheader("⚙️ Hyperparameter Tuning (SVM)")

        with st.expander("Run GridSearchCV"):
            c_values = st.multiselect(
                "Select C values",
                [0.1, 1, 10, 100],
                default=[1, 10]
            )

            kernel_values = st.multiselect(
                "Select kernels",
                ["linear", "rbf", "poly"],
                default=["linear", "rbf"]
            )

            gamma_values = st.multiselect(
                "Select gamma values (rbf/poly)",
                ["scale", "auto"],
                default=["scale"]
            )

            if st.button("Run Hyperparameter Tuning"):
                with st.spinner("Running GridSearchCV..."):
                    param_grid = {
                        "C": c_values,
                        "kernel": kernel_values,
                        "gamma": gamma_values
                    }

                    base_svm = SVC(decision_function_shape="ovr")

                    grid = GridSearchCV(
                        base_svm,
                        param_grid,
                        scoring="f1_weighted",
                        cv=3,
                        n_jobs=-1
                    )

                    grid.fit(X, y_true_encoded)

                    st.success("Tuning Completed ✅")

                    st.write("### ✅ Best Parameters")
                    st.json(grid.best_params_)

                    st.write(
                        f"### 🏆 Best CV F1 Score: "
                        f"**{grid.best_score_:.3f}**"
                    )

                    # Optional: use tuned model in app session
                    if st.checkbox("Use tuned model for predictions (session only)"):
                        svm_model = grid.best_estimator_
                        st.success("Tuned model activated for this session")

# ==========================================================
# INSIGHTS
# ==========================================================
elif menu == "Insights":
    st.title("🧠 Insights")
    if len(st.session_state.history) >= 3:
        avg = pd.DataFrame(st.session_state.history)["score"].mean()
        if avg < 0.35:
            st.error("⚠️ Strong negative trend detected.")
        elif avg < 0.65:
            st.warning("⚠️ Emotional instability detected.")
        else:
            st.success("✅ Positive emotional trend.")
    else:
        st.info("Not enough data.")

# ==========================================================
# HISTORY
# ==========================================================
elif menu == "History":
    st.dataframe(pd.DataFrame(st.session_state.history))

# ==========================================================
# ABOUT
# ==========================================================
elif menu == "About":
    st.markdown("""
    **Emotion Analytics from Journal Apps for Early Mental Health Detection**  
    Final Year Project using NLP & Machine Learning
    """)

st.markdown("---")
st.markdown("<center>🧠 Emotion Analytics Dashboard</center>", unsafe_allow_html=True)
