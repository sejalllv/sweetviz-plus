from .outlier_detector import (
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    combined_outlier_report,
)
from .quality_score import compute_quality_score
from .cleaning_recommender import generate_recommendations
from .auto_cleaner import auto_clean, before_after_comparison
from .report_builder import SweetVizPlusReport

__version__ = "0.1.0"


def analyze(df):
    return SweetVizPlusReport(df)
