import pandas as pd


def compute_quality_score(df: pd.DataFrame, outlier_summary: dict) -> dict:
    total_rows = len(df)
    total_cells = df.shape[0] * df.shape[1]

    total_missing_cells = df.isna().sum().sum()

    if total_cells > 0:
        missing_penalty = (total_missing_cells / total_cells) * 100 * 0.4
    else:
        missing_penalty = 0

    duplicate_rows = df.duplicated().sum()

    if total_rows > 0:
        duplicate_penalty = (duplicate_rows / total_rows) * 100 * 0.3
    else:
        duplicate_penalty = 0

    total_outlier_rows = len(outlier_summary.get("all_outlier_row_indexes", set()))

    if total_rows > 0:
        outlier_penalty = (total_outlier_rows / total_rows) * 100 * 0.3
    else:
        outlier_penalty = 0

    raw_score = 100 - missing_penalty - duplicate_penalty - outlier_penalty
    score = max(0, raw_score)

    grade = assign_grade(score)

    breakdown = {
        "missing_penalty": round(missing_penalty, 2),
        "duplicate_penalty": round(duplicate_penalty, 2),
        "outlier_penalty": round(outlier_penalty, 2),
        "total_missing_cells": int(total_missing_cells),
        "total_cells": int(total_cells),
        "duplicate_rows": int(duplicate_rows),
        "total_rows": int(total_rows),
        "total_outlier_rows": int(total_outlier_rows),
    }

    return {
        "score": round(score, 2),
        "grade": grade,
        "breakdown": breakdown,
    }


def assign_grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


if __name__ == "__main__":
    from .outlier_detector import combined_outlier_report

    print("Running self-test on sample_data/titanic_sample.csv ...\n")

    test_df = pd.read_csv("sample_data/titanic_sample.csv")
    outlier_summary = combined_outlier_report(test_df)
    result = compute_quality_score(test_df, outlier_summary)

    print("Quality Score:", result["score"])
    print("Grade:", result["grade"])
    print("Breakdown:")
    for key, value in result["breakdown"].items():
        print(f"  {key}: {value}")
