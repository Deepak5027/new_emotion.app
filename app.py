# ==========================================================
# MODEL EVALUATION (FINAL – GUARANTEED METRICS)
# ==========================================================
elif menu == "Model Evaluation":
    st.title("📊 Model Evaluation")

    uploaded_file = st.file_uploader(
        "Upload CSV file (text + label columns)",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("📌 Upload a labeled CSV file to evaluate the model.")
        st.stop()

    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # Column selection
    text_col = st.selectbox("Select TEXT column", df.columns)
    label_col = st.selectbox("Select LABEL column", df.columns)

    # Safety check
    if text_col == label_col:
        st.error("❌ Text column and Label column must be different")
        st.stop()

    # Clean text
    X = df[text_col].astype(str).apply(clean_text)

    # Normalize labels
    y = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    # Keep only labels known to model
    valid_labels = set(label_encoder.classes_)
    mask = y.isin(valid_labels)

    X = X[mask]
    y = y[mask]

    if len(y) < 5:
        st.error("❌ Not enough valid labeled samples to evaluate")
        st.stop()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Vectorize
    X_test_vec = vectorizer.transform(X_test)

    # Predict
    y_pred_encoded = svm_model.predict(X_test_vec)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # ================= METRICS =================
    st.subheader("📈 Performance Metrics")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # ================= REPORT =================
    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    # ================= CONFUSION MATRIX =================
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)

    fig, ax = plt.subplots(figsize=(6, 4))
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
    st.pyplot(fig)
