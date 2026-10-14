import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    outlier_rows = []
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_row_indexes = df.index[is_outlier]

        for row_index in outlier_row_indexes:
            outlier_rows.append({
                "column_name": col,
                "row_index": row_index,
                "value": df.loc[row_index, col],
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            })

    result_df = pd.DataFrame(
        outlier_rows,
        columns=["column_name", "row_index", "value", "lower_bound", "upper_bound"]
    )
    return result_df


def detect_outliers_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df_clean = numeric_df.dropna()

    if len(numeric_df_clean) == 0:
        return pd.DataFrame(columns=["row_index", "isolation_forest_flag"])

    model = IsolationForest(contamination=contamination, random_state=42)
    flags = model.fit_predict(numeric_df_clean)

    result_df = pd.DataFrame({
        "row_index": numeric_df_clean.index,
        "isolation_forest_flag": flags
    })
    return result_df


def combined_outlier_report(df: pd.DataFrame) -> dict:
    iqr_results = detect_outliers_iqr(df)
    isolation_forest_results = detect_outliers_isolation_forest(df)

    if len(iqr_results) > 0:
        summary_by_column = iqr_results.groupby("column_name").size().to_dict()
    else:
        summary_by_column = {}

    if len(isolation_forest_results) > 0:
        iso_outlier_rows = set(
            isolation_forest_results.loc[
                isolation_forest_results["isolation_forest_flag"] == -1, "row_index"
            ]
        )
    else:
        iso_outlier_rows = set()

    iqr_outlier_rows = set(iqr_results["row_index"]) if len(iqr_results) > 0 else set()

    flagged_by_both = iqr_outlier_rows & iso_outlier_rows
    flagged_by_iqr_only = iqr_outlier_rows - iso_outlier_rows
    flagged_by_iso_only = iso_outlier_rows - iqr_outlier_rows

    agreement = {
        "flagged_by_both": len(flagged_by_both),
        "flagged_by_iqr_only": len(flagged_by_iqr_only),
        "flagged_by_isolation_forest_only": len(flagged_by_iso_only),
    }

    all_outlier_row_indexes = iqr_outlier_rows | iso_outlier_rows

    return {
        "iqr_results": iqr_results,
        "isolation_forest_results": isolation_forest_results,
        "summary_by_column": summary_by_column,
        "isolation_forest_outlier_count": len(iso_outlier_rows),
        "agreement": agreement,
        "all_outlier_row_indexes": all_outlier_row_indexes,
    }


if __name__ == "__main__":
    print("Running self-test on sample_data/titanic_sample.csv ...\n")

    test_df = pd.read_csv("sample_data/titanic_sample.csv")
    report = combined_outlier_report(test_df)

    print("IQR outliers found (first 10 rows):")
    print(report["iqr_results"].head(10))
    print()

    print("IQR outlier count per column:")
    print(report["summary_by_column"])
    print()

    print("Isolation Forest flagged", report["isolation_forest_outlier_count"], "rows as outliers")
    print()

    print("Agreement between methods:")
    print(report["agreement"])
    print()

    print("Total unique rows flagged by EITHER method:", len(report["all_outlier_row_indexes"]))
