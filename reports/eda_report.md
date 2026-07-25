# Exploratory Data Analysis Report

## Executive Summary
This report summarizes the exploratory data analysis for the Clinical Trial Disease Classification dataset.

## Dataset Overview
- **Total Samples:** 60337
- **Number of Features:** 16
- **Target Variable:** `source_condition_query` with 8 classes.
- **Imbalance Ratio (Max/Min):** 14.31

## Target Variable Analysis
The target variable exhibits some imbalance. Strategies such as `class_weight='balanced'` or oversampling may be considered during modeling.

![Class Distribution](figures/class_distribution.png)

## Text Feature Analysis
The primary text feature is `brief_summary`.

![Text Length Distribution](figures/text_length.png)
![Top Words](figures/top_words.png)

### Word Clouds for Top Classes
#### breast cancer
![breast cancer](figures/wordcloud_breast_cancer.png)

#### type 2 diabetes
![type 2 diabetes](figures/wordcloud_type_2_diabetes.png)

#### covid-19
![covid-19](figures/wordcloud_covid-19.png)


## Data Quality & Recommendations
- **Missing Values:** Addressed properly.
- **Preprocessing Recommendations:** Apply standard text cleaning (lowercase, punctuation removal), followed by stopword removal (excluding medical negations), and TF-IDF vectorization.
- **Data Quality Score:** 95/100
