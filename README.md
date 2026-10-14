# SweetViz+

Extends the open-source [SweetViz](https://github.com/fbdesignpro/sweetviz) library with four things it doesn't do:

1. Outlier detection: IQR (per column) and Isolation Forest (whole rows)
2. A dataset quality score from 0 to 100, with a letter grade
3. Plain-language cleaning recommendations
4. Auto-clean of the safe issues, with a before/after score

SweetViz itself is used through its public API only and is not modified.

## Install

From GitHub (works in Jupyter, Colab and any terminal):

```
pip install git+https://github.com/sejalllv/sweetviz-plus.git
```

In a Colab cell, put an exclamation mark in front:

```
!pip install git+https://github.com/sejalllv/sweetviz-plus.git
```

To also run the Streamlit demo app, download this repository and run, from its folder:

```
pip install -e ".[app]"
streamlit run app.py
```

## Use it in a notebook

```python
import pandas as pd
import sweetviz_plus as svp

df = pd.read_csv("your_data.csv")

report = svp.analyze(df)
report.summary()

cleaned_df = report.auto_clean()
report.summary()

report.show_notebook()
```

Individual pieces are also available:

```python
report.quality
report.outliers
report.recommendations
report.comparison
```

A full walkthrough is in `SweetViz_Plus_Demo.ipynb`.

## Known limitations

- Isolation Forest assumes 5% of rows are outliers, so it always flags about 5% of the rows it can score.
- Isolation Forest skips any row with a missing numeric value.
- The quality score is a heuristic with fixed weights (0.4 missing, 0.3 duplicates, 0.3 outliers), not a validated measurement.
- Auto-clean only fixes safe issues: it drops columns that are over 70% missing, removes exact duplicate rows, and caps IQR outliers in columns that are not highly skewed. Missing values and skew are left for a human to decide.

## Credits and license

Built on [SweetViz](https://github.com/fbdesignpro/sweetviz) by fbdesignpro, which is MIT licensed. SweetViz+ is also released under the MIT License (see `LICENSE`).
