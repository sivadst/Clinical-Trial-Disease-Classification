class ClinicalClassifierError(Exception):
    """Base exception class for Clinical Trial Disease Classifier."""

    pass


class DataValidationError(ClinicalClassifierError):
    """Raised when data validation fails."""

    pass


class PreprocessingError(ClinicalClassifierError):
    """Raised when text preprocessing fails."""

    pass


class ModelTrainingError(ClinicalClassifierError):
    """Raised when model training fails."""

    pass


class ModelEvaluationError(ClinicalClassifierError):
    """Raised when model evaluation fails."""

    pass


class AppStartupError(ClinicalClassifierError):
    """Raised when the Streamlit app fails to start or load artifacts."""

    pass
