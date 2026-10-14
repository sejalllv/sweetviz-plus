import pandas as pd
from .outlier_detector import combined_outlier_report
from .quality_score import compute_quality_score


def auto_clean(df: pd.DataFrame, outlier_summary: dict) -> tuple:
    cleaned_df = df.copy()

    total_rows = len(cleaned_df)
    columns_to_drop = []
    for col in cleaned_df.columns:
        missing_pct = (cleaned_df[col].isna().sum() / total_rows) * 100 if total_rows > 0 else 0
        if missing_pct > 70:
            columns_to_drop.append(col)
    cleaned_df = cleaned_df.drop(columns=columns_to_drop)

    cleaned_df = cleaned_df.drop_duplicates()

    skipped_due_to_skew = []
    iqr_results = outlier_summary.get("iqr_results", pd.DataFrame())
    if len(iqr_results) > 0:
        bounds_by_column = iqr_results.groupby("column_name").agg(
            lower_bound=("lower_bound", "first"),
            upper_bound=("upper_bound", "first")
        )
        for col, bounds in bounds_by_column.iterrows():
            if col not in cleaned_df.columns:
                continue

            if cleaned_df[col].notna().sum() < 3:
                continue

            skewness = cleaned_df[col].skew()
            if skewness > 1 or skewness < -1:
                skipped_due_to_skew.append(col)
                continue

            cleaned_df[col] = cleaned_df[col].clip(
                lower=bounds["lower_bound"],
                upper=bounds["upper_bound"]
            )

    return cleaned_df, skipped_due_to_skew


def before_after_comparison(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, skipped_due_to_skew: list = None) -> dict:
    if skipped_due_to_skew is None:
        skipped_due_to_skew = []

    original_outlier_summary = combined_outlier_report(original_df)
    before_result = compute_quality_score(original_df, original_outlier_summary)

    cleaned_outlier_summary = combined_outlier_report(cleaned_df)
    after_result = compute_quality_score(cleaned_df, cleaned_outlier_summary)

    changes_made = []

    dropped_columns = set(original_df.columns) - set(cleaned_df.columns)
    if dropped_columns:
        changes_made.append(f"Dropped columns (over 70% missing): {', '.join(dropped_columns)}")

    rows_removed = len(original_df) - len(cleaned_df)
    if rows_removed > 0:
        changes_made.append(f"Removed {rows_removed} exact duplicate row(s)")

    changes_made.append("Capped IQR-flagged outliers to their column's normal range")

    if skipped_due_to_skew:
        changes_made.append(
            f"Did NOT cap outliers in {', '.join(skipped_due_to_skew)} — highly skewed, "
            f"needs human judgment instead of automatic clipping"
        )

    return {
        "before_score": before_result["score"],
        "before_grade": before_result["grade"],
        "after_score": after_result["score"],
        "after_grade": after_result["grade"],
        "changes_made": changes_made,
    }


if __name__ == "__main__":
    print("Running self-test on sample_data/titanic_sample.csv ...\n")

    test_df = pd.read_csv("sample_data/titanic_sample.csv")
    outlier_summary = combined_outlier_report(test_df)

    cleaned_df, skipped_due_to_skew = auto_clean(test_df, outlier_summary)
    comparison = before_after_comparison(test_df, cleaned_df, skipped_due_to_skew)

    print("BEFORE:", comparison["before_score"], "(" + comparison["before_grade"] + ")")
    print("AFTER: ", comparison["after_score"], "(" + comparison["after_grade"] + ")")
    print()
    print("Changes made:")
    for change in comparison["changes_made"]:
        print(" -", change)
    print()
    print("Original shape:", test_df.shape)
    print("Cleaned shape: ", cleaned_df.shape)
