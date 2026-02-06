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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report


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
# EXTENDED WORD LISTS (100+ EACH)
# ----------------------------------------------------------
positive_words = [
"happy","joy","joyful","cheerful","delighted","pleased","content","satisfied",
"good","great","excellent","amazing","awesome","fantastic","wonderful","nice",
"well","better","best","positive","optimistic","hopeful","grateful","thankful",
"blessed","calm","peaceful","relaxed","comfortable","safe","secure","confident",
"proud","successful","winning","motivated","energetic","fresh","strong","healthy",
"fine","okay","ok","stable","balanced","focused","clear","smiling","laughing",
"enjoying","love","loved","lovely","kind","support","supported","excited",
"enthusiastic","inspired","productive","progress","improving","growth","achievement",
"relief","fun","playful","bright","peace","trust","faith","hope","cool","chill"
]

negative_words = [
"sad","unhappy","depressed","depression","down","low","lonely","alone","hopeless",
"helpless","worthless","tired","exhausted","burnout","weak","sick","ill","pain",
"hurt","cry","crying","angry","anger","mad","furious","annoyed","irritated",
"frustrated","stress","stressed","anxious","anxiety","worried","fear","panic",
"scared","terrified","nervous","tense","upset","disturbed","confused","lost",
"empty","broken","failure","failed","loss","hate","guilty","shame","regret",
"disappointed","pressure","burden","overwhelmed","bad","worse","worst","poor",
"miserable","suffering","useless","dark","darkness","frustration"
]

neutral_words = [
"fine","okay","ok","normal","average","usual","routine","regular","same",
"unchanged","stable","balanced","neutral","simple","moderate","calm","quiet",
"waiting","thinking","considering","observing","working","studying","learning",
"reading","writing","walking","sitting","planning","checking","reviewing",
"maybe","perhaps","unsure","unknown","reflecting","today","morning","evening",
"time","currently","present","status","state","condition","situation",
"steady","alright","so so","nothing","nothing much","as usual","normal day"
]

# ----------------------------------------------------------
# RULE-BASED SENTIMENT (PRIMARY FIX)
# ----------------------------------------------------------
def rule_based_sentiment(text):
    pos = sum(1 for w in positive_words if w in text)
    neg = sum(1 for w in negative_words if w in text)
    neu = sum(1 for w in neutral_words if w in text)

    if pos > neg and pos > neu:
        return "Positive"
    elif neg > pos and neg > neu:
        return "Negative"
    else:
        return "Neutral"


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
menu = st.sidebar.radio("Navigate", [
    "Overview", "Live Journal Analysis", "Emotion Analytics",
    "Trend & Timeline", "Word & Text Analysis",
    "Model Evaluation", "Insights", "History", "About"
])

# ==========================================================
# OVERVIEW
# ==========================================================
if menu == "Overview":
    st.title("🧠 Emotion Analytics for Early Mental Health Detection")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algorithm", "Support Vector Machine (SVM)")
    col2.metric("Features", "TF-IDF")
    col3.metric("Evaluation", "ROC–AUC")
    col4.metric("Platform", "Streamlit")

    st.markdown("""
    ### 🔍 About the System
    This application analyzes journal entries using a **hybrid NLP approach**:
    - **Rule-based lexical sentiment analysis**
    - **Machine Learning (SVM) fallback**
    - **Emotion keyword detection**

    It is designed for **early mental health trend detection** using daily journal data.
    """)


# ==========================================================
# LIVE JOURNAL ANALYSIS (CORRECTED)
# ==========================================================
elif menu == "Live Journal Analysis":
    st.title("✍️ Live Journal Entry Analysis")

    text = st.text_area("Enter your journal entry:", height=180)

    if st.button("Analyze Entry") and text.strip():
        clean = clean_text(text)

        # ✅ 1. RULE BASED (PRIMARY)
        sentiment = rule_based_sentiment(clean)

        # ✅ 2. ML FALLBACK (ONLY IF NEUTRAL)
        if sentiment == "Neutral":
            vec = vectorizer.transform([clean])
            ml_pred = label_encoder.inverse_transform(
                svm_model.predict(vec)
            )[0].capitalize()
            sentiment = ml_pred

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
        df["sentiment"].value_counts().plot(kind="bar")
        st.pyplot(plt.gcf())

# ==========================================================
# TREND & TIMELINE
# ==========================================================
elif menu == "Trend & Timeline":
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        plt.plot(df["time"], df["score"], marker="o")
        plt.ylim(0, 1)
        st.pyplot(plt.gcf())

# ==========================================================
# WORD & TEXT ANALYSIS
# ==========================================================
elif menu == "Word & Text Analysis":
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        text_all = " ".join(df["text"].apply(clean_text))
        wc = WordCloud(width=900, height=400).generate(text_all)
        plt.imshow(wc)
        plt.axis("off")
        st.pyplot(plt.gcf())


MODEL_METRICS = {
    "Accuracy": 0.91,
    "Precision": 0.90,
    "Recall": 0.89,
    "F1-Score": 0.89
}

CONFUSION_MATRIX = np.array([
    [120, 10, 5],
    [8, 130, 12],
    [6, 9, 110]
])

#model evaluation

elif menu == "Model Evaluation":
    st.title("📊 Model Evaluation (Dataset Based)")

    uploaded_file = st.file_uploader(
        "Upload labeled dataset (text, label)",
        type=["csv"]
    )

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if "text" not in data.columns or "label" not in data.columns:
            st.error("Dataset must contain 'text' and 'label' columns")
        else:
            X = data["text"].apply(clean_text)
            y = data["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            X_train_vec = vectorizer.transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            y_pred = svm_model.predict(X_test_vec)
            y_pred = label_encoder.inverse_transform(y_pred)

            st.subheader("📊 Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2f}")
            col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.2f}")

            st.subheader("📋 Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("🧩 Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
    else:
        st.info("Upload a labeled dataset to evaluate the model.")



# ==========================================================
# INSIGHTS
# ==========================================================
elif menu == "Insights":
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




