import streamlit as st
import pandas as pd
import sweetviz as sv

from sweetviz_plus.outlier_detector import combined_outlier_report
from sweetviz_plus.quality_score import compute_quality_score
from sweetviz_plus.cleaning_recommender import generate_recommendations
from sweetviz_plus.auto_cleaner import auto_clean, before_after_comparison

st.set_page_config(page_title="SweetViz+", layout="wide")

st.title("SweetViz+")
st.write("Upload a CSV or use the sample Titanic dataset to see outlier detection, a quality score, cleaning recommendations, and an automatic clean-up with before/after comparison.")

if "df" not in st.session_state:
    st.session_state.df = None
if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None
if "clean_result" not in st.session_state:
    st.session_state.clean_result = None
if "sweetviz_html" not in st.session_state:
    st.session_state.sweetviz_html = None


def reset_saved_results():
    st.session_state.clean_result = None
    st.session_state.sweetviz_html = None


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

use_sample = st.button("Use sample Titanic dataset instead")

if use_sample:
    st.session_state.df = pd.read_csv("sample_data/titanic_sample.csv")
    reset_saved_results()
elif uploaded_file is not None:
    upload_id = (uploaded_file.name, uploaded_file.size)
    if upload_id != st.session_state.last_upload_id:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.last_upload_id = upload_id
        reset_saved_results()

df = st.session_state.df

if df is not None:
    st.header("1. Dataset Preview")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10))

    with st.spinner("Running outlier detection..."):
        outlier_summary = combined_outlier_report(df)

    with st.spinner("Computing quality score..."):
        quality_result = compute_quality_score(df, outlier_summary)

    with st.spinner("Generating recommendations..."):
        recommendations = generate_recommendations(df, outlier_summary, quality_result["breakdown"])

    st.header("2. Outlier Detection")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("IQR outliers per column")
        if outlier_summary["summary_by_column"]:
            st.write(outlier_summary["summary_by_column"])
        else:
            st.write("No IQR outliers found.")

    with col2:
        st.subheader("Isolation Forest")
        st.write(f"Flagged {outlier_summary['isolation_forest_outlier_count']} row(s) as multivariate outliers")
        st.write("Agreement between methods:")
        st.write(outlier_summary["agreement"])

    if len(outlier_summary["iqr_results"]) > 0:
        st.subheader("All flagged values (IQR)")
        st.dataframe(outlier_summary["iqr_results"])

    st.header("3. Quality Score")

    score_col1, score_col2 = st.columns(2)
    with score_col1:
        st.metric("Quality Score", f"{quality_result['score']} / 100")
    with score_col2:
        st.metric("Grade", quality_result["grade"])

    st.write("Penalty breakdown:")
    st.write(quality_result["breakdown"])

    st.header("4. Cleaning Recommendations")
    if recommendations:
        for rec in recommendations:
            st.write("- " + rec)
    else:
        st.write("No recommendations — dataset looks clean.")

    st.header("5. Auto-Clean")
    if st.button("Run Auto-Clean"):
        with st.spinner("Cleaning dataset..."):
            cleaned_df, skipped_due_to_skew = auto_clean(df, outlier_summary)
            comparison = before_after_comparison(df, cleaned_df, skipped_due_to_skew)
        st.session_state.clean_result = {
            "cleaned_df": cleaned_df,
            "comparison": comparison,
        }

    if st.session_state.clean_result is not None:
        cleaned_df = st.session_state.clean_result["cleaned_df"]
        comparison = st.session_state.clean_result["comparison"]

        before_col, arrow_col, after_col = st.columns([2, 1, 2])
        with before_col:
            st.metric("Before", f"{comparison['before_score']} ({comparison['before_grade']})")
        with arrow_col:
            st.markdown("<h1 style='text-align: center;'>→</h1>", unsafe_allow_html=True)
        with after_col:
            st.metric("After", f"{comparison['after_score']} ({comparison['after_grade']})")

        st.write("Changes made:")
        for change in comparison["changes_made"]:
            st.write("- " + change)

        st.write(f"Shape before: {df.shape} → Shape after: {cleaned_df.shape}")

        st.subheader("Cleaned data preview")
        st.dataframe(cleaned_df.head(10))

        csv_download = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned CSV", csv_download, "cleaned_data.csv", "text/csv")

    st.header("6. Full SweetViz Report")
    if st.button("Generate SweetViz Report"):
        with st.spinner("Generating SweetViz report (this may take a moment)..."):
            report = sv.analyze(df)
            report.show_html("sweetviz_report.html", open_browser=False, layout="vertical")
            with open("sweetviz_report.html", "r", encoding="utf-8") as f:
                st.session_state.sweetviz_html = f.read()

    if st.session_state.sweetviz_html is not None:
        st.success("Report generated below, and also saved as sweetviz_report.html in your project folder.")
        st.download_button("Download SweetViz Report (HTML)", st.session_state.sweetviz_html, "sweetviz_report.html", "text/html")
        st.components.v1.html(st.session_state.sweetviz_html, height=800, scrolling=True)

else:
    st.info("Upload a CSV or click the sample dataset button above to get started.")
