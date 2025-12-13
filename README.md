# 🧠 Mental Health Journal Sentiment Analysis

This project focuses on analyzing mental health journal entries to classify user sentiment using machine learning. The goal is to assist in early mental health detection by identifying emotional patterns in written journal data.

---

## 📌 Project Overview

Mental health journals contain rich emotional information. By applying Natural Language Processing (NLP) techniques and machine learning models, this project classifies journal entries into sentiment categories such as positive, negative, or neutral.

The project uses a **Linear Support Vector Machine (SVM)** model trained on preprocessed journal text data and deployed as an interactive application.

---

## 🎯 Objectives

- Preprocess and clean mental health journal text data
- Convert text into numerical features using TF-IDF
- Train and evaluate a Linear SVM sentiment classifier
- Visualize model results and accuracy
- Deploy the trained model for real-world use

---

## 🧪 Dataset

- **Source:** Mental health journal entries (anonymized)
- **Size:** ~70,000 records
- **Key Columns:**
  - `journal_text` – User journal entry
  - `sentiment_label` – Sentiment category
  - `sentiment_score` – Numeric sentiment score
  - `word_count` – Length of journal entry

---

## ⚙️ Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib
- **Model:** Linear Support Vector Machine (SVM)
- **Feature Extraction:** TF-IDF Vectorizer
- **Deployment:** Streamlit + GitHub + Streamlit Cloud

---

## 🧠 Machine Learning Pipeline

1. **Text Preprocessing**
   - Lowercasing
   - Removing special characters
   - Cleaning noise

2. **Feature Extraction**
   - TF-IDF Vectorization
   - Unigrams and bigrams

3. **Model Training**
   - Linear SVM classifier

4. **Evaluation**
   - Accuracy
   - Class-wise recall
   - Prediction analysis

---

## 📊 Results

- The final Linear SVM model achieved reliable accuracy on unseen test data.
- The model performed better than Logistic Regression.
- Visual analysis confirms consistent prediction behavior.

**Final Model Accuracy:**  
> Displayed directly in the application.

---

## 🚀 Deployment

The trained model components were saved using Joblib and deployed using Streamlit.

### Saved Files:
- `svm_sentiment_model.joblib`
- `tfidf_vectorizer.joblib`
- `label_encoder.joblib`

The application loads these files at runtime to make real-time predictions.

---

## ▶️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mental-health-sentiment-analysis.git
