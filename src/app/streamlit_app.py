import time
import os
import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Ensure the root directory is in the sys path for relative imports when running from elsewhere
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import config
from src.data.preprocessor import TextPreprocessor

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Clinical Trial Disease Category Classifier",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# --- Master Data for 8 Target Disease Categories ---
DISEASE_CATEGORIES = [
    {"name": "Breast Cancer", "key": "breast cancer", "icon": "🎗️", "default_count": 16301, "desc": "Oncology clinical trials focused on breast carcinomas and therapies."},
    {"name": "Type 2 Diabetes", "key": "type 2 diabetes", "icon": "🩸", "default_count": 11467, "desc": "Endocrine studies evaluating insulin, glycemic control, and metabolic health."},
    {"name": "COVID-19", "key": "covid-19", "icon": "🦠", "default_count": 10153, "desc": "Infectious disease trials investigating SARS-CoV-2 treatments and vaccines."},
    {"name": "Anxiety", "key": "anxiety", "icon": "🧠", "default_count": 9286, "desc": "Psychiatric trials studying generalized anxiety, panic disorders, and CBT."},
    {"name": "COPD", "key": "chronic obstructive pulmonary disease", "icon": "🫁", "default_count": 6181, "desc": "Pulmonology studies on chronic obstructive pulmonary disease and bronchodilators."},
    {"name": "Rheumatoid Arthritis", "key": "rheumatoid arthritis", "icon": "🦴", "default_count": 3637, "desc": "Autoimmune & rheumatology research targeting joint inflammation and biologics."},
    {"name": "Glaucoma", "key": "glaucoma", "icon": "👁️", "default_count": 2173, "desc": "Ophthalmology clinical trials for intraocular pressure and optic nerve protection."},
    {"name": "Sickle Cell Anemia", "key": "sickle cell anemia", "icon": "🔬", "default_count": 1139, "desc": "Hematology trials focused on hemoglobin disorders and gene therapies."},
]

# --- 5 Clickable Sample Clinical Trial Summaries for Instant Testing ---
SAMPLE_SUMMARIES = {
    "🎗️ Breast Cancer": (
        "A Randomized, Double-Blind Phase III Trial Comparing Trastuzumab Plus Chemotherapy "
        "With Chemotherapy Alone in Patients With HER2-Overexpressing Metastatic Breast Cancer. "
        "Primary endpoint is overall survival and progression-free survival in HER2 positive invasive ductal carcinoma patients."
    ),
    "🩸 Type 2 Diabetes": (
        "Efficacy and Safety of Metformin Monotherapy versus Sitagliptin in Patients with Inadequately "
        "Controlled Type 2 Diabetes Mellitus. Evaluation of HbA1c levels, fasting plasma glucose, insulin resistance, "
        "and body mass index over 24 weeks."
    ),
    "🦠 COVID-19": (
        "A Multicenter Clinical Trial Evaluating the Efficacy of Remdesivir in Hospitalized Adults "
        "Diagnosed with Severe COVID-19 Pneumonia. Assessment of time to clinical recovery, oxygen saturation levels, "
        "mechanically ventilated status, and 28-day mortality."
    ),
    "🧠 Anxiety": (
        "Cognitive Behavioral Therapy versus Selective Serotonin Reuptake Inhibitors for Generalized "
        "Anxiety Disorder. Evaluation of Hamilton Anxiety Rating Scale (HAM-A) scores, panic symptoms, "
        "emotional regulation, and quality of life in young adults."
    ),
    "🫁 COPD": (
        "A Double-Blind Study of Tiotropium Bromide Inhalation Powder versus Placebo in Patients "
        "with Moderate to Severe Chronic Obstructive Pulmonary Disease (COPD). Primary outcomes measure Forced Expiratory "
        "Volume in 1 second (FEV1), exacerbation frequency, and dyspnea score."
    ),
}

# --- Caching Artifacts & Data ---
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


GDRIVE_FILE_ID = "1JpRmEoQEk2l7NzJ5bc5vaC7NR2rgOKwX"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"


@st.cache_data(show_spinner=False)
def load_data():
    # 1. Try local full raw file if > 1000 rows
    if os.path.exists(config.RAW_DATA_PATH):
        try:
            df = pd.read_csv(config.RAW_DATA_PATH)
            if len(df) > 1000:
                return df
        except Exception:
            pass

    # 2. Try previously downloaded full dataset from Google Drive
    downloaded_csv_path = os.path.join(config.BASE_DIR, "data", "raw", "full_gdrive_dataset.csv")
    if os.path.exists(downloaded_csv_path) and os.path.getsize(downloaded_csv_path) > 1000000:
        try:
            df = pd.read_csv(downloaded_csv_path)
            if len(df) > 1000:
                return df
        except Exception:
            pass

    # 3. Try multi-category JSON sample data (800 rows across all 8 classes)
    json_sample_path = os.path.join(config.BASE_DIR, "data", "raw", "clinical_trials_sample.json")
    if os.path.exists(json_sample_path):
        try:
            df = pd.read_json(json_sample_path)
            if not df.empty:
                return df
        except Exception:
            pass

    return pd.DataFrame()


def download_gdrive_dataset():
    downloaded_csv_path = os.path.join(config.BASE_DIR, "data", "raw", "full_gdrive_dataset.csv")
    with st.spinner("Downloading full 60,337 clinical trials dataset from Google Drive (158 MB)..."):
        try:
            import gdown
            os.makedirs(os.path.dirname(downloaded_csv_path), exist_ok=True)
            gdown.download(GDRIVE_URL, downloaded_csv_path, quiet=False)
            st.cache_data.clear()
            st.success("Successfully downloaded full dataset! Please refresh the page.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to download Google Drive dataset: {e}")


# Initialize core objects
model, vectorizer, encoder = load_artifacts()
df = load_data()
preprocessor = TextPreprocessor()


def render_footer():
    st.write("---")
    st.markdown(
        "<div style='text-align: center; color: #888888; padding: 15px; font-size: 0.9em;'>"
        "Built by <b>Selvasiva S</b> | GUVI Zen Data Science Career Program | "
        "<b>Clinical Trial Disease Category Classification Using NLP & Machine Learning</b>"
        "</div>",
        unsafe_allow_html=True
    )


# --- Navigation Sidebar ---
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Select View",
    [
        "Home / Dashboard",
        "Data Explorer",
        "EDA Visualizations",
        "Model Performance",
        "Prediction",
        "Batch Prediction",
        "About / Capstone Info",
    ],
)

st.sidebar.write("---")
st.sidebar.info(
    "**GUVI Zen Data Science Capstone**\n\n"
    "**Author:** Selvasiva S\n\n"
    "**Target Classes:** 8 Disease Categories\n\n"
    "**Best Model:** LinearSVC (F1: 0.9436)"
)

# ====================================================
# 1. HOME / DASHBOARD PAGE
# ====================================================
if page == "Home / Dashboard":
    st.title("🏥 Clinical Trial Disease Category Classification")
    st.caption("GUVI Zen Data Science Capstone Project | Production NLP & ML Pipeline")

    st.markdown("""
    ### 🏠 Project Overview
    Welcome to the **Clinical Trial Disease Category Classifier**. This application provides a production-grade 
    Natural Language Processing (NLP) and Machine Learning pipeline that automatically analyzes unstructured clinical trial 
    brief summaries from **ClinicalTrials.gov** and categorizes them into **8 distinct medical disease areas**.
    """)

    # High-level Metric Cards
    st.subheader("📊 Key Project Metrics")
    col1, col2, col3, col4 = st.columns(4)

    is_full_data = not df.empty and len(df) > 1000
    total_records = len(df) if not df.empty else 60337

    with col1:
        st.metric("Total Records", f"{total_records:,}", delta="Full Dataset" if is_full_data else "Training Dataset")
    with col2:
        st.metric("Disease Categories", "8 Classes", delta="Target Classes")
    with col3:
        st.metric("Best Model", "LinearSVC", delta="Calibrated")
    with col4:
        st.metric("Accuracy / F1-Score", "95.19% / 0.9436", delta="Held-out Test Set")

    if not is_full_data:
        with st.expander("ℹ️ Dataset Status & Remote Access", expanded=False):
            st.info(
                "Displaying statistics from the cached multi-category sample dataset. "
                "Click below to fetch the full **60,337 record** dataset directly from Google Drive."
            )
            if st.button("📥 Load Full 60,337 Dataset from Google Drive", type="secondary"):
                download_gdrive_dataset()

    st.write("---")

    # --- 🩺 Supported Disease Categories Section ---
    st.subheader("🩺 Supported Disease Categories")
    st.caption("The model classifies clinical trial summaries into these disease categories.")

    val_counts = {}
    if not df.empty and config.TARGET_COLUMN in df.columns:
        val_counts = df[config.TARGET_COLUMN].astype(str).str.lower().value_counts().to_dict()

    cols = st.columns(4)
    for idx, cat in enumerate(DISEASE_CATEGORIES):
        col = cols[idx % 4]
        key_lower = cat["key"].lower()
        count = val_counts.get(key_lower, cat["default_count"])

        with col:
            with st.container(border=True):
                st.markdown(f"**{cat['icon']} {cat['name']}**")
                st.caption(cat["desc"])
                st.markdown(f"📊 **{count:,}** samples")

    st.write("---")

    st.markdown("""
    ### 🏗️ Application Features
    - **🤖 Real-Time Text Classification**: Input clinical descriptions to predict disease categories instantly.
    - **💡 5 Instant Test Samples**: Click pre-loaded clinical summaries to evaluate model inference speed.
    - **🔍 Interactive Data Explorer**: Search, filter, and analyze clinical trial text across categories.
    - **📈 Model Performance Benchmarks**: Explore cross-validation metrics, confusion matrices, and F1 charts.
    - **📦 Bulk Batch Prediction**: Upload CSV files for automated batch text categorization.
    """)

    render_footer()

# ====================================================
# 2. DATA EXPLORER PAGE
# ====================================================
elif page == "Data Explorer":
    st.header("📊 Interactive Data Explorer")
    st.caption("Explore, search, filter, and inspect clinical trial records across all 8 disease categories.")

    if df.empty:
        st.info("The sample dataset is initializing. Displaying summary benchmarks across target categories.")
    else:
        st.write(f"Displaying **{len(df):,}** available clinical trial records.")

        tab1, tab2, tab3 = st.tabs(["🔍 Search & Filter Data", "📊 Dataset Statistics", "🔤 Text Analysis"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                search_text = st.text_input("🔍 Search Clinical Summary Text", "")
            with col2:
                all_classes = [cat["key"] for cat in DISEASE_CATEGORIES]
                present_classes = sorted(df[config.TARGET_COLUMN].dropna().unique().tolist()) if config.TARGET_COLUMN in df.columns else all_classes
                selected_classes = st.multiselect("📂 Filter by Disease Category", options=present_classes)

            filtered_df = df.copy()
            if search_text:
                filtered_df = filtered_df[
                    filtered_df[config.TEXT_COLUMN].astype(str).str.contains(search_text, case=False, na=False)
                ]
            if selected_classes and config.TARGET_COLUMN in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[config.TARGET_COLUMN].isin(selected_classes)]

            st.markdown(f"Showing **{len(filtered_df):,}** of **{len(df):,}** matching records.")
            st.dataframe(filtered_df, use_container_width=True)

            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Filtered Data (CSV)",
                data=csv,
                file_name="filtered_clinical_trials.csv",
                mime="text/csv",
            )

        with tab2:
            st.subheader("📊 Dataset Statistics & Class Distribution")
            
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            with s_col1:
                st.metric("Total Records", f"{len(filtered_df):,}")
            with s_col2:
                st.metric("Unique Classes", f"{filtered_df[config.TARGET_COLUMN].nunique() if config.TARGET_COLUMN in filtered_df.columns else 8}")
            with s_col3:
                missing_c = filtered_df[config.TEXT_COLUMN].isnull().sum() if config.TEXT_COLUMN in filtered_df.columns else 0
                st.metric("Missing Summaries", f"{missing_c}")
            with s_col4:
                avg_w = int(filtered_df[config.TEXT_COLUMN].astype(str).str.split().str.len().mean()) if config.TEXT_COLUMN in filtered_df.columns else 98
                st.metric("Avg Words / Summary", f"{avg_w}")

            if config.TARGET_COLUMN in filtered_df.columns and not filtered_df.empty:
                st.write("---")
                st.markdown("#### Class Frequency Distribution")
                class_counts = filtered_df[config.TARGET_COLUMN].value_counts()
                st.bar_chart(class_counts)

        with tab3:
            st.subheader("🔤 Text Length & Word Frequency Analysis")
            if config.TEXT_COLUMN in filtered_df.columns and not filtered_df.empty:
                word_counts = filtered_df[config.TEXT_COLUMN].astype(str).str.split().str.len()
                st.markdown("#### Summary Word Count Distribution")
                st.line_chart(word_counts.value_counts().sort_index())
            else:
                st.info("Text length distribution chart available when dataset is loaded.")

    render_footer()

# ====================================================
# 3. EDA VISUALIZATIONS PAGE
# ====================================================
elif page == "EDA Visualizations":
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.caption("Visual insights generated during data inspection and feature analysis.")

    figures = {
        "Class Distribution across Disease Categories": "class_distribution.png",
        "Missing Values Heatmap": "missing_values.png",
        "Clinical Summary Text Length Distribution": "text_length.png",
        "Top Most Frequent Words in Clinical Summaries": "top_words.png",
        "Word Cloud — Breast Cancer": "wordcloud_breast_cancer.png",
        "Word Cloud — COVID-19": "wordcloud_covid-19.png",
        "Word Cloud — Type 2 Diabetes": "wordcloud_type_2_diabetes.png",
    }

    for title, filename in figures.items():
        st.subheader(title)
        img_path = os.path.join(config.FIGURES_DIR, filename)
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption=title, use_container_width=True)
        else:
            st.info(f"Figure `{filename}` will appear here upon pipeline execution.")

    render_footer()

# ====================================================
# 4. MODEL PERFORMANCE PAGE
# ====================================================
elif page == "Model Performance":
    st.header("📈 Model Performance & Evaluation")
    st.caption("Comprehensive benchmarking metrics evaluated on the 20% held-out test set.")

    # High-level Metrics Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Best Model", "LinearSVC", delta="Calibrated")
    with col2:
        st.metric("Test Accuracy", "95.19%", delta="+0.83% vs XGBoost")
    with col3:
        st.metric("F1-Score (Macro)", "0.9436", delta="Primary Metric")
    with col4:
        st.metric("Precision (Macro)", "94.80%", delta="Macro Average")
    with col5:
        st.metric("5-Fold CV Score", "94.10%", delta="Cross-Validated")

    st.write("---")

    # Benchmark Comparison Table
    st.subheader("🏆 Classifier Benchmarking Comparison")
    comparison_data = pd.DataFrame([
        {"Model": "LinearSVC (Calibrated)", "Accuracy": "95.19%", "F1 Macro": "0.9436", "Precision": "94.80%", "Recall": "94.20%", "Status": "🥇 Best Model"},
        {"Model": "XGBoost Classifier", "Accuracy": "94.90%", "F1 Macro": "0.9414", "Precision": "94.50%", "Recall": "93.90%", "Status": "🥈 Runner-up"},
        {"Model": "Logistic Regression", "Accuracy": "94.71%", "F1 Macro": "0.9391", "Precision": "94.10%", "Recall": "93.80%", "Status": "🥉 Baseline Top"},
        {"Model": "Random Forest", "Accuracy": "94.32%", "F1 Macro": "0.9365", "Precision": "93.90%", "Recall": "93.40%", "Status": "Tree Ensemble"},
        {"Model": "Multinomial Naive Bayes", "Accuracy": "90.59%", "F1 Macro": "0.8934", "Precision": "89.20%", "Recall": "89.50%", "Status": "Fast Baseline"},
    ])
    st.dataframe(comparison_data, use_container_width=True)

    st.write("---")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.subheader("Model F1-Score Comparison")
        comp_img = os.path.join(config.FIGURES_DIR, "model_comparison_f1.png")
        if os.path.exists(comp_img):
            st.image(Image.open(comp_img), caption="F1 Macro Score Comparison across Models", use_container_width=True)
        else:
            st.info("Comparison chart available upon pipeline execution.")

    with col_fig2:
        st.subheader("LinearSVC Confusion Matrix")
        cm_img = os.path.join(config.FIGURES_DIR, "confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(Image.open(cm_img), caption="Confusion Matrix of the Best LinearSVC Model", use_container_width=True)
        else:
            st.info("Confusion matrix available upon pipeline execution.")

    render_footer()

# ====================================================
# 5. PREDICTION PAGE
# ====================================================
elif page == "Prediction":
    st.header("🔬 Live Disease Category Prediction")
    st.markdown("Input a clinical trial brief summary below or click one of the quick test samples to predict its disease category.")

    # State management for sample selection
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

    def select_sample(sample_text):
        st.session_state["user_input"] = sample_text

    # Sample Buttons
    st.subheader("💡 Click a Sample to Test Instantly:")
    s_cols = st.columns(5)
    sample_items = list(SAMPLE_SUMMARIES.items())
    for idx, (label, text) in enumerate(sample_items):
        with s_cols[idx]:
            if st.button(label, use_container_width=True):
                select_sample(text)

    user_input = st.text_area(
        "Clinical Trial Summary",
        value=st.session_state["user_input"],
        height=180,
        placeholder="Type or paste a clinical trial brief summary here...",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        classify_btn = st.button("🔍 Classify Disease", type="primary", use_container_width=True)

    if classify_btn:
        if not user_input or not user_input.strip():
            st.warning("⚠️ Please enter or select a clinical summary before classifying.")
        elif model is None or vectorizer is None or encoder is None:
            st.error("❌ Model artifacts are not loaded. Run `python scripts/train.py` first.")
        else:
            start_time = time.time()
            with st.spinner("Processing clinical text through NLP pipeline..."):
                try:
                    preprocessed = preprocessor.preprocess(user_input)
                    if not preprocessed:
                        st.warning("⚠️ Text is too short or contains only non-informative terms after preprocessing.")
                    else:
                        vectorized = vectorizer.transform([preprocessed])
                        prediction = model.predict(vectorized)[0]
                        probabilities = model.predict_proba(vectorized)[0]
                        predicted_class = encoder.inverse_transform([prediction])[0]
                        elapsed_ms = (time.time() - start_time) * 1000
                        max_prob = float(max(probabilities))

                        st.write("---")
                        st.subheader("🎯 Prediction Output & Analysis")

                        col_res1, col_res2, col_res3 = st.columns(3)
                        with col_res1:
                            st.success(f"### ✅ Predicted Disease:\n**{predicted_class.title()}**")
                        with col_res2:
                            st.metric("📊 Confidence Score", f"{max_prob:.2%}")
                            st.progress(max_prob)
                        with col_res3:
                            st.metric("⏱ Prediction Time", f"{elapsed_ms:.1f} ms")

                        # Confidence Explanation Card
                        with st.container(border=True):
                            if max_prob >= 0.80:
                                st.markdown("📄 **Confidence Interpretation**: **High Confidence (>80%)**. The clinical summary contains distinct disease-specific medical terminology.")
                            elif max_prob >= 0.50:
                                st.markdown("📄 **Confidence Interpretation**: **Moderate Confidence (50-80%)**. The clinical text shares terminology common across multiple related medical conditions.")
                            else:
                                st.markdown("📄 **Confidence Interpretation**: **Low Confidence (<50%)**. High ambiguity in trial description.")

                        st.write("---")
                        st.subheader("📈 Probability Distribution Across All 8 Disease Categories")

                        all_classes = encoder.classes_
                        prob_df = pd.DataFrame({
                            "Disease Category": [c.title() for c in all_classes],
                            "Probability": probabilities
                        }).sort_values(by="Probability", ascending=False)

                        st.bar_chart(prob_df.set_index("Disease Category"))
                        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Classification failed: {str(e)}")

    render_footer()

# ====================================================
# 6. BATCH PREDICTION PAGE
# ====================================================
elif page == "Batch Prediction":
    st.header("📦 Automated Batch Prediction")
    st.write("Upload a CSV file containing clinical trial summaries to categorize all records in bulk.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("📄 Uploaded Data Preview:")
            st.dataframe(batch_df.head(), use_container_width=True)

            text_cols = [c for c in batch_df.columns if batch_df[c].dtype == "object"]
            if not text_cols:
                st.error("No text columns found in the CSV.")
            else:
                col_to_predict = st.selectbox("Select column containing clinical summary text:", text_cols)

                if st.button("🚀 Run Batch Classification", type="primary", use_container_width=True):
                    if model is None:
                        st.error("Models not loaded.")
                    else:
                        with st.spinner("Processing batch records..."):
                            processed = batch_df[col_to_predict].astype(str).apply(preprocessor.preprocess)
                            vecs = vectorizer.transform(processed)
                            preds = model.predict(vecs)
                            pred_labels = encoder.inverse_transform(preds)
                            probs = model.predict_proba(vecs)
                            max_probs = np.max(probs, axis=1)

                            batch_df["Predicted_Category"] = [p.title() for p in pred_labels]
                            batch_df["Confidence"] = max_probs

                            st.success("✅ Batch Prediction Complete!")
                            st.dataframe(batch_df, use_container_width=True)

                            csv = batch_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Download Predictions (CSV)",
                                data=csv,
                                file_name="batch_clinical_predictions.csv",
                                mime="text/csv",
                            )
        except Exception as e:
            st.error(f"Error reading file: {e}")

    render_footer()

# ====================================================
# 7. ABOUT PAGE
# ====================================================
elif page == "About / Capstone Info":
    st.header("📄 Capstone Project & Developer Information")

    st.markdown("""
    ### 🏥 Clinical Trial Disease Category Classification Using NLP and Machine Learning

    **Project Description**:  
    This capstone project implements an end-to-end Machine Learning solution to automatically structure and classify 
    unstructured clinical trial brief summaries from [ClinicalTrials.gov](https://clinicaltrials.gov/) into **8 distinct medical disease areas**.

    ---

    ### 🎯 Key Objectives
    1. **Automated NLP Preprocessing**: Custom cleaning pipeline that expands medical abbreviations (`pt` → `patient`, `dx` → `diagnosis`) and explicitly preserves medical negations (`no`, `not`, `denies`).
    2. **Multi-Model Benchmarking**: Trains and evaluates 5 classifiers (MultinomialNB, Logistic Regression, RandomForest, LinearSVC, XGBoost).
    3. **Imbalance-Aware Evaluation**: Utilizes **F1-Macro** to penalize majority-class bias across all 8 target disease categories.
    4. **Production Deployment**: Serves real-time and batch predictions via this interactive multi-page Streamlit application.

    ---

    ### 🛠️ Tech Stack & Architecture
    - **Language**: Python 3.9+
    - **Natural Language Processing**: NLTK, Scikit-learn TF-IDF Vectorizer
    - **Machine Learning**: Scikit-Learn (LinearSVC, LogisticRegression, RandomForest, MultinomialNB), XGBoost
    - **Data Manipulation**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn, WordCloud
    - **Web Framework**: Streamlit
    - **Configuration**: Pydantic BaseSettings

    ---

    ### 👤 Developer Information
    - **Developer**: Selvasiva S
    - **Program**: GUVI Zen Data Science Career Program
    - **Capstone Domain**: Natural Language Processing & Machine Learning
    - **GitHub Repository**: [github.com/sivadst/Clinical-Trial-Disease-Classification](https://github.com/sivadst/Clinical-Trial-Disease-Classification)

    ---

    ### 🔮 Future Enhancements
    - Integrate Transformer-based models (BioBERT, ClinicalBERT) for fine-grained clinical entity recognition.
    - Support multi-label classification for trials addressing co-morbidities.
    - Deploy containerized REST API endpoints using FastAPI and Docker.
    """)

    render_footer()
