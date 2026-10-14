import pandas as pd

from .outlier_detector import combined_outlier_report
from .quality_score import compute_quality_score
from .cleaning_recommender import generate_recommendations
from .auto_cleaner import auto_clean, before_after_comparison


class SweetVizPlusReport:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.outliers = combined_outlier_report(df)
        self.quality = compute_quality_score(df, self.outliers)
        self.recommendations = generate_recommendations(df, self.outliers, self.quality["breakdown"])
        self.cleaned_df = None
        self.comparison = None

    def auto_clean(self) -> pd.DataFrame:
        cleaned_df, skipped_due_to_skew = auto_clean(self.df, self.outliers)
        self.cleaned_df = cleaned_df
        self.comparison = before_after_comparison(self.df, cleaned_df, skipped_due_to_skew)
        return cleaned_df

    def summary(self):
        breakdown = self.quality["breakdown"]
        print(f"Rows: {breakdown['total_rows']}   Columns: {self.df.shape[1]}")
        print(f"Quality score: {self.quality['score']} / 100   (Grade {self.quality['grade']})")
        print()
        print("Outliers")
        print("  IQR outliers per column:", self.outliers["summary_by_column"] or "none")
        print("  Rows flagged by Isolation Forest:", self.outliers["isolation_forest_outlier_count"])
        print("  Agreement between methods:", self.outliers["agreement"])
        print()
        print("Recommendations")
        if self.recommendations:
            for number, rec in enumerate(self.recommendations, start=1):
                print(f"  {number}. {rec}")
        else:
            print("  None - dataset looks clean.")
        if self.comparison is not None:
            print()
            print("Auto-clean")
            print(
                f"  {self.comparison['before_score']} ({self.comparison['before_grade']})"
                f" -> {self.comparison['after_score']} ({self.comparison['after_grade']})"
            )
            for change in self.comparison["changes_made"]:
                print("  -", change)

    def show_notebook(self, **kwargs):
        import sweetviz as sv
        return sv.analyze(self.df).show_notebook(**kwargs)

    def show_html(self, filepath: str = "sweetviz_report.html", open_browser: bool = False, layout: str = "vertical"):
        import sweetviz as sv
        sv.analyze(self.df).show_html(filepath, open_browser=open_browser, layout=layout)
        return filepath
