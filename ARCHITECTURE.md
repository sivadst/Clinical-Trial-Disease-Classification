# Architecture Decision Record (ADR)

## Context and Problem Statement
We need to build an automated classification pipeline to determine disease categories from clinical trial brief summaries. The system must be robust, production-grade, highly cohesive, loosely coupled, and simple enough for stakeholders to view dynamically via an interface.

## Technology Stack
- **Python 3.9+**: For maximum compatibility and robust typing systems.
- **Pandas / NumPy**: For rapid, vector-based data manipulations and transformations.
- **Scikit-Learn / XGBoost**: For classical Machine Learning and Tree ensemble capabilities. Chosen due to their excellent balance of inference speed and classification metrics compared to heavier Deep Learning frameworks.
- **NLTK**: For fundamental text operations like tokenization, lemmatization, and stopword mapping.
- **Streamlit**: For rapid UI prototyping and interactive data dashboards.
- **Pytest**: For rigorous integration and unit testing.

## Architectural Design Patterns
- **Strategy Pattern (Models)**: By abstracting `BaseClassifier`, we guarantee that any algorithm (Naive Bayes, SVM, XGBoost) honors the same contract (`train`, `predict`, `evaluate`, `save`, `load`). The `ModelFactory` instantiates these generically.
- **Template / Pipeline Pattern (Data)**: The `TextPreprocessor` serves as a consistent sequence of deterministic string manipulations ensuring text looks identical at training and inference.
- **Singleton Pattern (Logger / Config)**: Standardizing our `get_logger` ensures thread-safe, consistent log streams without duplication. The application state relies entirely on a centralized `Settings` class instantiated once.

## Key Decisions

### Decision 1: TF-IDF over Dense Embeddings
**Status:** Accepted
**Rationale:** Given the relatively low complexity of predicting 8 overarching disease classes and the need for computational efficiency (running on standard hardware without GPUs), a capped TF-IDF vectorizer (max 10,000 features, bigrams) provides strong, robust performance without the computational overhead of BERT-like models. F1 metrics (>0.94) validate this choice.

### Decision 2: Streamlit for Deployment UI
**Status:** Accepted
**Rationale:** Medical researchers require intuitive dashboards. Streamlit allows pure Python implementations of interactive applications with simple deployment pathways, drastically reducing the frontend engineering burden.

### Decision 3: F1-Macro as Primary Metric
**Status:** Accepted
**Rationale:** The dataset suffers from distinct class imbalances (e.g., thousands of Breast Cancer trials vs far fewer Sickle Cell Anemia). Accuracy can mask poor performance on minority classes. F1-Macro forces the optimizer and evaluator to treat every class with equal importance.

## Future Enhancements
- Expand to multi-label classification to support trials covering multiple distinct conditions.
- Replace TF-IDF with Bio-Clinical BERT if finer granularity prediction (e.g., rare diseases) becomes necessary.
- Containerize with Docker for strict reproducible environments.
