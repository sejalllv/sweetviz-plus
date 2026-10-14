import pandas as pd


def generate_recommendations(df: pd.DataFrame, outlier_summary: dict, quality_breakdown: dict) -> list:
    recommendations = []
    total_rows = len(df)

    for col in df.columns:
        missing_count = df[col].isna().sum()
        if total_rows == 0:
            continue
        missing_pct = (missing_count / total_rows) * 100

        if missing_pct > 50:
            recommendations.append(
                f"Column '{col}' is {missing_pct:.1f}% missing — "
                f"consider dropping it or imputing missing values."
            )
        elif missing_pct >= 5:
            recommendations.append(
                f"Column '{col}' has {missing_pct:.1f}% missing values — "
                f"consider imputation (mean/median/mode) before analysis."
            )

    duplicate_rows = quality_breakdown.get("duplicate_rows", df.duplicated().sum())
    if total_rows > 0:
        duplicate_pct = (duplicate_rows / total_rows) * 100
    else:
        duplicate_pct = 0

    if duplicate_pct > 5:
        recommendations.append(
            f"{duplicate_rows} duplicate rows detected ({duplicate_pct:.1f}% of dataset) — "
            f"consider removing duplicates."
        )

    summary_by_column = outlier_summary.get("summary_by_column", {})
    for col, outlier_count in summary_by_column.items():
        non_missing_count = df[col].notna().sum()
        if non_missing_count == 0:
            continue
        outlier_pct_in_column = (outlier_count / non_missing_count) * 100

        if outlier_pct_in_column > 10:
            recommendations.append(
                f"Column '{col}' has a high concentration of outliers — "
                f"consider reviewing them or applying a transformation (e.g., log scale)."
            )

    numeric_columns = df.select_dtypes(include="number").columns
    for col in numeric_columns:
        if df[col].notna().sum() < 3:
            continue
        skewness = df[col].skew()

        if skewness > 1 or skewness < -1:
            recommendations.append(
                f"Column '{col}' is highly skewed — "
                f"consider a log or square-root transform before modeling."
            )

    return recommendations


if __name__ == "__main__":
    from .outlier_detector import combined_outlier_report
    from .quality_score import compute_quality_score

    print("Running self-test on sample_data/titanic_sample.csv ...\n")

    test_df = pd.read_csv("sample_data/titanic_sample.csv")
    outlier_summary = combined_outlier_report(test_df)
    quality_result = compute_quality_score(test_df, outlier_summary)

    recs = generate_recommendations(test_df, outlier_summary, quality_result["breakdown"])

    print(f"Generated {len(recs)} recommendation(s):\n")
    for i, rec in enumerate(recs, start=1):
        print(f"{i}. {rec}")
