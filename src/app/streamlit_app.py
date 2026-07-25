import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Ensure the root directory is in the sys path for relative imports when running from elsewhere


from config.settings import config
from src.data.preprocessor import TextPreprocessor

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Clinical Trial Disease Classifier", page_icon="🏥")


# --- Caching Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(os.path.join(config.MODELS_DIR, "best_model.pkl"))
        vectorizer = joblib.load(os.path.join(config.MODELS_DIR, "tfidf_vectorizer.pkl"))
        encoder = joblib.load(os.path.join(config.MODELS_DIR, "label_encoder.pkl"))
        return model, vectorizer, encoder
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return None, None, None


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()


# Initialize objects
model, vectorizer, encoder = load_artifacts()
df = load_data()
preprocessor = TextPreprocessor()


# --- Navigation ---
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home / Overview",
        "Data Explorer",
        "EDA Visualizations",
        "Model Performance",
        "Prediction",
        "Batch Prediction",
        "About / Documentation",
    ],
)

# --- Pages ---

if page == "EDA Visualizations":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Below are some visual insights generated from the dataset during the EDA phase.")

    figures = {
        "Class Distribution": "class_distribution.png",
        "Missing Values Heatmap": "missing_values.png",
        "Text Length Distribution": "text_length.png",
        "Top Words": "top_words.png",
        "Breast Cancer Wordcloud": "wordcloud_breast_cancer.png",
        "COVID-19 Wordcloud": "wordcloud_covid-19.png",
        "Type 2 Diabetes Wordcloud": "wordcloud_type_2_diabetes.png",
    }

    for title, filename in figures.items():
        st.subheader(title)
        img_path = os.path.join(config.FIGURES_DIR, filename)
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption=title)
        else:
            st.warning(f"Image not found: {filename}")

elif page == "Home / Overview":
    st.title("🏥 Clinical Trial Disease Category Classifier")
    st.markdown("""
    This application uses Natural Language Processing and Machine Learning
    to automatically classify clinical trial summaries into disease categories.

    ### Capabilities
    - **Automatic Text Classification**: Input a clinical summary and receive instant disease category prediction.
    - **Interactive Data Explorer**: Browse and filter the clinical trial dataset.
    - **Model Performance**: Explore performance metrics of the best model.
    - **Production-Grade Pipeline**: Built with robust preprocessing and validated ML models.
    """)

    if not df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Disease Categories", f"{df[config.TARGET_COLUMN].nunique()}")
        with col3:
            st.metric("Features (Raw)", f"{len(df.columns)}")

    st.markdown("""
    ### System Architecture
    * **Data Layer:** Pandas loading & parsing from `clinical_trials_raw_patient2trial_conditions.csv`
    * **Processing Layer:** TF-IDF with medical abbreviation expansion & custom Stopword handling
    * **Model Layer:** Sklearn/XGBoost ensemble & classical algorithms
    * **Serving Layer:** Streamlit (this app)
    """)

elif page == "Data Explorer":
    st.header("📊 Interactive Data Explorer")
    if df.empty:
        st.warning("Data is not available.")
    else:
        st.write("Browse the underlying dataset used to train the model.")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            search_text = st.text_input("🔍 Search Summary Text", "")
        with col2:
            selected_classes = st.multiselect(
                "📂 Filter by Disease Class",
                options=sorted(df[config.TARGET_COLUMN].dropna().unique()),
            )

        filtered_df = df.copy()
        if search_text:
            filtered_df = filtered_df[
                filtered_df[config.TEXT_COLUMN].str.contains(search_text, case=False, na=False)
            ]
        if selected_classes:
            filtered_df = filtered_df[filtered_df[config.TARGET_COLUMN].isin(selected_classes)]

        st.write(f"Showing {len(filtered_df):,} of {len(df):,} trials.")
        st.dataframe(filtered_df, use_container_width=True)

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered Data",
            data=csv,
            file_name="filtered_clinical_trials.csv",
            mime="text/csv",
        )

elif page == "Model Performance":
    st.header("📈 Model Performance")

    # Check if report exists
    report_path = os.path.join(config.REPORTS_DIR, "model_report.md")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report_md = f.read()

        # Extract images from markdown text and display via st.image if possible, or just raw markdown
        # Quick hack: we can display the text, but images in markdown might not render locally well without path adjustment.
        # We'll just show the images explicitly.

        st.subheader("Model Comparison Metrics")
        try:
            comparison_img = os.path.join(config.FIGURES_DIR, "model_comparison_f1.png")
            if os.path.exists(comparison_img):
                st.image(Image.open(comparison_img), caption="F1 Score Comparison across Models")
        except Exception:
            pass

        st.subheader("Best Model Confusion Matrix")
        try:
            cm_img = os.path.join(config.FIGURES_DIR, "confusion_matrix.png")
            if os.path.exists(cm_img):
                st.image(Image.open(cm_img), caption="Confusion Matrix of the Best Model")
        except Exception:
            pass
    else:
        st.info("Model report not found. Train the models first using `python scripts/train.py`.")

elif page == "Prediction":
    st.header("🔬 Disease Classification")
    st.markdown("Enter a clinical trial summary below to predict the disease category.")

    user_input = st.text_area(
        "Clinical Summary",
        height=200,
        placeholder="Enter the brief summary or description of the clinical trial...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
    with col2:
        pass  # placeholder for spacing or clear logic

    if classify_btn:
        if not user_input or not user_input.strip():
            st.error("⚠️ Please enter a clinical summary before classifying.")
        elif model is None or vectorizer is None or encoder is None:
            st.error("❌ Model artifacts are not loaded. Cannot predict.")
        else:
            with st.spinner("Analyzing clinical text..."):
                try:
                    preprocessed = preprocessor.preprocess(user_input)
                    if not preprocessed:
                        st.warning(
                            "⚠️ Text is too short or contains only stopwords/symbols after preprocessing."
                        )
                    else:
                        vectorized = vectorizer.transform([preprocessed])

                        prediction = model.predict(vectorized)[0]
                        probabilities = model.predict_proba(vectorized)[0]

                        predicted_class = encoder.inverse_transform([prediction])[0]

                        # Top-3 predictions
                        top_3_idx = np.argsort(probabilities)[-3:][::-1]
                        top_3_classes = encoder.inverse_transform(top_3_idx)
                        top_3_probs = probabilities[top_3_idx]

                        st.success(f"### Predicted Category: **{predicted_class}**")
                        st.progress(float(max(probabilities)))
                        st.caption(f"Confidence: {max(probabilities):.2%}")

                        st.subheader("Top 3 Predictions")
                        for cls, prob in zip(top_3_classes, top_3_probs):
                            st.write(f"- **{cls}**: {prob:.2%}")

                except Exception as e:
                    st.error(f"❌ Classification failed: {str(e)}")

elif page == "Batch Prediction":
    st.header("📦 Batch Prediction")
    st.write("Upload a CSV file containing a column with clinical summaries to get predictions.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(batch_df.head(), use_container_width=True)

            text_cols = [c for c in batch_df.columns if batch_df[c].dtype == "object"]
            if not text_cols:
                st.error("No text columns found in the CSV.")
            else:
                col_to_predict = st.selectbox("Select column containing clinical text:", text_cols)

                if st.button("Run Batch Prediction", type="primary"):
                    if model is None:
                        st.error("Models not loaded.")
                    else:
                        with st.spinner("Processing batch..."):
                            processed = (
                                batch_df[col_to_predict].astype(str).apply(preprocessor.preprocess)
                            )
                            vecs = vectorizer.transform(processed)
                            preds = model.predict(vecs)
                            pred_labels = encoder.inverse_transform(preds)

                            # Probabilities
                            probs = model.predict_proba(vecs)
                            max_probs = np.max(probs, axis=1)

                            batch_df["Predicted_Class"] = pred_labels
                            batch_df["Confidence"] = max_probs

                            st.success("Batch Prediction Complete!")
                            st.dataframe(batch_df, use_container_width=True)

                            csv = batch_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Download Predictions",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                            )
        except Exception as e:
            st.error(f"Error reading file: {e}")


elif page == "About / Documentation":
    st.header("📄 About & Documentation")

    st.markdown("""
    ### 🔬 Clinical Trial Disease Category Classifier

    **Goal**: To build an autonomous, robust ML pipeline that correctly categorizes clinical trial brief summaries into distinct disease areas.

    ### ⚙️ Preprocessing Pipeline
    1. **Text Cleaning**: Lowercase conversion, special character removal.
    2. **Medical Abbreviations**: Expanding standard abbreviations (e.g., `pt` -> `patient`, `dx` -> `diagnosis`).
    3. **Stopword Removal**: Eliminating non-informative English words while **preserving negations** (`no`, `not`, `denies`) to maintain clinical meaning.
    4. **Lemmatization**: Converting words to their root forms.
    5. **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency extraction up to 10,000 features and unigram/bigram spans.

    ### 🧠 Modeling Strategy
    - Stratified splits (70/10/20 train/val/test).
    - Class weight balancing natively to handle dataset imbalance.
    - Multiple base models including Naive Bayes, Logistic Regression, RandomForest, LinearSVC, and XGBoost.
    - Evaluation via **F1-Macro** score to penalize majority-class bias.

    ### 📊 Performance Metrics Glossary
    - **Accuracy**: Total correct predictions / Total predictions
    - **Precision**: How many of the predicted class X were actually class X.
    - **Recall**: How many of the actual class X were successfully found by the model.
    - **F1 Score**: Harmonic mean of Precision and Recall. F1-Macro averages the score equally across all classes, regardless of frequency.
    """)

    st.info("Developed as an end-to-end autonomous machine learning pipeline demonstration.")
