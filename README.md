# DataReady

## 🚀 Live Demo
👉 https://dataready-jntbkbxpzsudwabniq4imq.streamlit.app

Professional data analysis web app by **ASX Labs** — *Clean Data. Better Models.*

## Quick start

```bash
cd DataReady
pip install -r requirements.txt
streamlit run app.py
````

Open the URL shown in the terminal ([https://dataready-jntbkbxpzsudwabniq4imq.streamlit.app/](https://dataready-jntbkbxpzsudwabniq4imq.streamlit.app/)).

## Features

* **CSV upload** — Drag and drop; see row count, column count, file size
* **Data quality score** — 0–100 based on missing values, duplicates, type consistency
* **Missing values** — Per-column counts, percentages, bar chart
* **Outlier detection** — IQR-based; counts and chart by column
* **Feature correlation** — Heatmap and strongly correlated pairs with short explanations
* **Feature importance** — Random Forest importance; choose target column, see ranked chart
* **Recommendations** — Plain English suggestions for cleaning and modeling

## New Features Added

* **Data Type Profiler** — Shows column name, pandas dtype, inferred type, missing %, unique values, and flags possible issues
* **Cleaning Actions + Preview** — Toggle actions like removing duplicates, imputing missing values, capping outliers, dropping constant columns; preview before vs after counts
* **Export Cleaned Data + Report** — Download cleaned CSV and text report with dataset size, quality score, correlations, and recommendations

## Important Behavior Change

After these additions, **all analytics now run on the cleaned dataset** (based on your selected cleaning toggles), not just raw upload data.
Sections affected:

* Data Quality Score
* Missing Values
* Outlier Detection
* Correlation
* Feature Importance
* Recommendations

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, matplotlib, seaborn, SciPy.

## Branding

* Dark theme `#0F172A`
* Cyan accent `#13D8D8`
* ASX Labs logo and tagline in header
