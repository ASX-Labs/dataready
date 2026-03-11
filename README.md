# DataReady

Professional data analysis web app by **ASX Labs** — *Better Choices Start Here*.

## Quick start

```bash
cd DataReady
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (https://dataready-jntbkbxpzsudwabniq4imq.streamlit.app/).

## Features

- **CSV upload** — Drag and drop; see row count, column count, file size
- **Data quality score** — 0–100 based on missing values, duplicates, type consistency
- **Missing values** — Per-column counts, percentages, bar chart
- **Outlier detection** — IQR-based; counts and chart by column
- **Feature correlation** — Heatmap and strongly correlated pairs with short explanations
- **Feature importance** — Random Forest importance; choose target column, see ranked chart
- **Recommendations** — Plain English suggestions for cleaning and modeling

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, matplotlib, seaborn, SciPy.

## Branding

- Dark theme `#0F172A`
- Cyan accent `#13D8D8`
- ASX Labs logo and tagline in header
