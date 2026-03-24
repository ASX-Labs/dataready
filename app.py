"""
DataReady - Professional Data Analysis Web App
A product of ASX Labs | Clean Data. Better Models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from typing import Optional, Tuple

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="DataReady | ASX Labs",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============== CUSTOM CSS - ASX LABS BRANDING ==============
BRAND_BG = "#000000"
ACCENT = "#2ECC71"
ACCENT_DIM = "#2ECC71"
CARD_BG = "#1E293B"
TEXT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    .stApp {{
        background: {BRAND_BG};
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Hide default Streamlit branding */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    
    /* ASX Labs header */
    .brand-header {{
        background: linear-gradient(135deg, {BRAND_BG} 0%, #1a1a1a 100%);
        padding: 1rem 0 1.5rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 1px solid rgba(46, 204, 113, 0.2);
    }}
    .brand-logo {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {ACCENT};
        letter-spacing: 0.02em;
        margin-bottom: 0.25rem;
    }}
    .brand-tagline {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        font-weight: 400;
    }}
    
    /* Upload area */
    .upload-zone {{
        background: {CARD_BG};
        border: 2px dashed {ACCENT};
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        color: {TEXT_MUTED};
        transition: all 0.2s ease;
    }}
    .upload-zone:hover {{
        border-color: {ACCENT_DIM};
        background: rgba(30, 41, 59, 0.8);
    }}
    
    /* Score card */
    .score-card {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(46, 204, 113, 0.15);
    }}
    .score-value {{
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }}
    .score-label {{
        font-size: 0.9rem;
        color: {TEXT_MUTED};
        margin-top: 0.5rem;
    }}
    
    /* Section cards */
    .section-card {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(46, 204, 113, 0.1);
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background: {CARD_BG};
        border-radius: 8px;
    }}
    
    /* Buttons and inputs - dark theme overrides */
    .stButton > button {{
        background: {ACCENT} !important;
        color: {BRAND_BG} !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    .stButton > button:hover {{
        background: {ACCENT_DIM} !important;
        color: white !important;
    }}
    
    /* Dataframe styling */
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
    }}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {{
        color: {ACCENT};
        font-weight: 600;
    }}
    
    /* Recommendations box */
    .recommendations-box {{
        background: linear-gradient(180deg, {CARD_BG} 0%, #000000 100%);
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


def inject_dark_theme_matplotlib():
    """Set matplotlib to use dark theme colors for consistency."""
    plt.rcParams.update({
        "figure.facecolor": BRAND_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": ACCENT,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
    })


def render_header():
    """Render ASX Labs branding header."""
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">ASX Labs</div>
        <div class="brand-tagline">Clean Data. Better Models.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("## 📊 DataReady")
    st.markdown("*Clean Data. Better Models.*")
    st.markdown("Upload a CSV to analyze data quality, detect issues, and get actionable recommendations.")
    st.markdown("---")


def load_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load CSV and return (df, error_message).
    Auto-detects separator by trying comma, semicolon, then tab
    and picking the option that yields the most columns.
    """
    try:
        raw = uploaded_file.getvalue()
        if not raw:
            return None, "The file appears to be empty."

        text = raw.decode("utf-8", errors="replace")
        seps = [",", ";", "\t"]
        best_df: Optional[pd.DataFrame] = None
        best_cols = -1

        for sep in seps:
            try:
                df_try = pd.read_csv(StringIO(text), sep=sep)
                if df_try is not None and not df_try.empty:
                    n_cols = df_try.shape[1]
                    # Prefer earlier separators when tied, so only replace on strictly greater
                    if n_cols > best_cols:
                        best_cols = n_cols
                        best_df = df_try
            except Exception:
                continue

        # Fallback: let pandas infer if all attempts failed
        if best_df is None:
            try:
                best_df = pd.read_csv(StringIO(text), sep=None, engine="python")
            except Exception:
                best_df = None

        if best_df is None or best_df.empty:
            return None, "Could not detect a valid CSV separator. Try saving the file as a standard CSV with commas."

        return best_df, None
    except pd.errors.EmptyDataError:
        return None, "The file is empty or has no valid data."
    except pd.errors.ParserError as e:
        return None, f"Could not parse CSV: {str(e)[:200]}"
    except Exception as e:
        return None, f"Error reading file: {str(e)[:200]}"


def compute_quality_score(df: pd.DataFrame) -> Tuple[int, str]:
    """
    Compute overall data quality score 0-100.
    Based on: missing values, duplicates, data type consistency.
    Returns (score, color_hex).
    """
    n_rows, n_cols = df.shape
    if n_rows == 0:
        return 0, "#EF4444"

    # 1. Missing values penalty (up to 40 points)
    missing_pct = df.isna().sum().sum() / (n_rows * n_cols) * 100
    missing_score = max(0, 40 - (missing_pct * 0.4))  # 0% missing = 40 pts

    # 2. Duplicate rows penalty (up to 30 points)
    dup_pct = (df.duplicated().sum() / n_rows) * 100
    dup_score = max(0, 30 - (dup_pct * 0.3))

    # 3. Data type consistency (up to 30 points) - columns that are all same type
    type_score = 0
    for col in df.columns:
        col_missing = df[col].isna().sum()
        valid = df[col].dropna()
        if len(valid) == 0:
            type_score += 30 / len(df.columns)  # all missing, neutral
        else:
            # Check if column is numeric and has mixed types when read as object
            if df[col].dtype == "object":
                try:
                    pd.to_numeric(valid, errors="raise")
                    type_score += 30 / len(df.columns)
                except (ValueError, TypeError):
                    type_score += 30 / len(df.columns)  # object is consistent
            else:
                type_score += 30 / len(df.columns)

    total = missing_score + dup_score + min(30, type_score)
    score = int(round(min(100, max(0, total))))

    if score >= 70:
        color = "#22C55E"  # green
    elif score >= 40:
        color = "#EAB308"  # yellow
    else:
        color = "#EF4444"  # red

    return score, color


def get_numerical_columns(df: pd.DataFrame) -> list:
    """Return list of column names that are numeric."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def detect_outliers_iqr(series: pd.Series) -> np.ndarray:
    """Return boolean mask where True = outlier (IQR method)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        return np.zeros(len(series), dtype=bool)
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)


def feature_importance_rf(
    df: pd.DataFrame, target_col: str
) -> Tuple[Optional[pd.Series], Optional[float], Optional[str]]:
    """
    Compute feature importance using Random Forest.
    Returns:
    - feature importance series
    - model performance score on holdout set
    - metric name ("Accuracy" or "R²")
    """
    num_cols = get_numerical_columns(df)
    if target_col in num_cols:
        num_cols = [c for c in num_cols if c != target_col]
    else:
        num_cols = [c for c in num_cols if c != target_col]

    if not num_cols:
        return None, None, None

    X = df[num_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())

    y = df[target_col]
    metric_name = None
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        n_classes = len(le.classes_)
        if n_classes < 2:
            return None, None, None
        # Stratify for classification when possible.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        metric_name = "Accuracy"
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y[valid_idx]
        if len(y) < 10:
            return None, None, None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        metric_name = "R²"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    performance = (
        accuracy_score(y_test, y_pred)
        if metric_name == "Accuracy"
        else r2_score(y_test, y_pred)
    )
    imp = pd.Series(model.feature_importances_, index=num_cols).sort_values(ascending=True)
    return imp, float(performance), metric_name


def generate_recommendations(df: pd.DataFrame, quality_score: int,
                            missing_by_col: pd.Series, outlier_counts: dict,
                            has_correlations: bool) -> list:
    """Generate plain English recommendations."""
    recs = []

    if quality_score < 70:
        recs.append("Your overall data quality score is below 70. Focus on fixing missing values and duplicates first before building any models.")

    missing_cols = missing_by_col[missing_by_col > 0]
    if len(missing_cols) > 0:
        worst = missing_cols.idxmax()
        pct = missing_by_col[worst]
        recs.append(f"**Missing values:** Column '{worst}' has {pct:.1f}% missing values. Consider imputing (e.g. median for numbers, mode for categories) or dropping the column if it's not essential.")

    if df.duplicated().sum() > 0:
        recs.append(f"**Duplicates:** You have {df.duplicated().sum()} duplicate rows. Remove them with pandas: `df.drop_duplicates(inplace=True)` unless duplicates are meaningful.")

    if outlier_counts:
        high_outlier_cols = [c for c, n in outlier_counts.items() if n > 0]
        if high_outlier_cols:
            recs.append(f"**Outliers:** These columns have outliers (IQR method): {', '.join(high_outlier_cols[:5])}{'...' if len(high_outlier_cols) > 5 else ''}. Decide whether to cap, remove, or keep them based on domain knowledge.")

    if has_correlations:
        recs.append("**Correlation:** You have strongly correlated features. If you use regularized models or care about interpretability, consider dropping one from each highly correlated pair to reduce redundancy.")

    if not recs:
        recs.append("Your data looks in good shape for modeling. Do a final check: ensure your target column has no data leakage (e.g. IDs or future info) and that train/test split is appropriate.")

    return recs


def build_data_type_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Build a simple profile of each column and potential issues."""
    rows = []
    for col in df.columns:
        s = df[col]
        missing_pct = float(s.isna().mean() * 100)
        unique_vals = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        inferred = "categorical/text"

        if np.issubdtype(s.dtype, np.number):
            inferred = "numeric"
        elif np.issubdtype(s.dtype, np.datetime64):
            inferred = "datetime"
        elif dtype == "object":
            non_null = s.dropna().astype(str)
            as_num = pd.to_numeric(non_null.str.replace(",", ""), errors="coerce")
            num_ratio = as_num.notna().mean() if len(non_null) else 0
            if num_ratio >= 0.9:
                inferred = "numeric (stored as text)"
            else:
                as_dt = pd.to_datetime(non_null, errors="coerce")
                dt_ratio = as_dt.notna().mean() if len(non_null) else 0
                if dt_ratio >= 0.9:
                    inferred = "datetime (stored as text)"

        flag = ""
        if "id" in col.lower() and unique_vals >= max(1, int(len(df) * 0.8)):
            flag = "possible ID"
        elif unique_vals <= 1:
            flag = "constant/near-constant"
        elif missing_pct > 30:
            flag = "high missingness"

        rows.append(
            {
                "Column": col,
                "Pandas dtype": dtype,
                "Inferred type": inferred,
                "Missing %": round(missing_pct, 1),
                "Unique values": unique_vals,
                "Flag": flag,
            }
        )
    return pd.DataFrame(rows)


def apply_cleaning_actions(
    df: pd.DataFrame,
    remove_duplicates: bool,
    impute_missing: bool,
    cap_outliers: bool,
    drop_constant: bool,
) -> pd.DataFrame:
    """Apply selected cleaning actions and return a cleaned copy."""
    cleaned = df.copy()

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    if drop_constant:
        nunique = cleaned.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            cleaned = cleaned.drop(columns=constant_cols)

    if impute_missing:
        for col in cleaned.columns:
            if cleaned[col].isna().sum() == 0:
                continue
            if np.issubdtype(cleaned[col].dtype, np.number):
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            else:
                mode_values = cleaned[col].mode(dropna=True)
                fill_value = mode_values.iloc[0] if not mode_values.empty else np.nan
                cleaned[col] = cleaned[col].fillna(fill_value)

    if cap_outliers:
        for col in cleaned.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)

    return cleaned


@st.cache_data(show_spinner=False)
def compute_outlier_counts_cached(df: pd.DataFrame) -> dict:
    """Compute outlier counts per numeric column using IQR; cached for speed."""
    counts = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        mask = detect_outliers_iqr(df[col].dropna())
        counts[col] = int(mask.sum())
    return counts


def build_report_text(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    quality_score: int,
    recommendations: list,
    strong_pairs: list,
) -> str:
    """Generate a plain text analysis report."""
    dup_count = int(original_df.duplicated().sum())
    missing_pct = (cleaned_df.isna().sum() / max(1, len(cleaned_df)) * 100).sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(3)

    lines = [
        "DataReady Analysis Report",
        "ASX Labs | Clean Data. Better Models.",
        "",
        "Dataset Overview",
        f"- Original rows: {len(original_df):,}",
        f"- Original columns: {original_df.shape[1]:,}",
        f"- Cleaned rows: {len(cleaned_df):,}",
        f"- Cleaned columns: {cleaned_df.shape[1]:,}",
        f"- Data quality score: {quality_score}/100",
        f"- Duplicate rows (original): {dup_count}",
        "",
        "Top Issues",
    ]

    if len(top_missing) > 0:
        for col, pct in top_missing.items():
            lines.append(f"- {pct:.1f}% missing in '{col}'")
    else:
        lines.append("- No remaining missing values in cleaned data")

    lines.extend([
        "",
        "Strong Correlation Pairs (|r| >= 0.6)",
    ])

    if strong_pairs:
        for a, b, r in strong_pairs[:15]:
            lines.append(f"- {a} <-> {b}: {r:.2f}")
    else:
        lines.append("- None detected")

    lines.append("")
    lines.append("Recommendations")
    for rec in recommendations:
        lines.append(f"- {rec.replace('**', '')}")

    return "\n".join(lines)


def main():
    inject_dark_theme_matplotlib()
    render_header()

    # ============== CSV UPLOAD ==============
    st.markdown("### 1. Upload your data")
    uploaded_file = st.file_uploader(
        "Drag and drop a CSV file here, or click to browse",
        type=["csv"],
        help="Only CSV files are supported.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to get started. We'll show you data quality, outliers, correlations, and recommendations.")
        return

    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File too large (max 10MB). Please upload a smaller CSV.")
        return

    with st.spinner("Loading your data..."):
        df, err = load_csv(uploaded_file)

    if err:
        st.error(f"**Could not load file:** {err}")
        return

    # Basic info
    n_rows, n_cols = df.shape
    file_size_kb = uploaded_file.size / 1024
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{n_rows:,}")
    with c2:
        st.metric("Columns", f"{n_cols:,}")
    with c3:
        st.metric("File size", f"{file_size_kb:.1f} KB")

    st.markdown("---")

    with st.expander("**2. Data Type Profiler**", expanded=False):
        profile_df = build_data_type_profile(df)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        flagged = profile_df[profile_df["Flag"] != ""]
        if not flagged.empty:
            st.caption("Flagged columns may need cleanup before modeling.")

    with st.expander("**3. Data Cleaning Actions + Preview**", expanded=True):
        if st.button("Auto Clean (Recommended)", use_container_width=True):
            st.session_state["remove_duplicates"] = True
            st.session_state["impute_missing"] = True
            st.session_state["cap_outliers"] = False
            st.session_state["drop_constant"] = True
            st.success("Auto Clean options applied.")

        col_a, col_b = st.columns(2)
        with col_a:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True, key="remove_duplicates")
            impute_missing = st.checkbox("Impute missing values (median/mode)", value=True, key="impute_missing")
        with col_b:
            cap_outliers = st.checkbox("Cap numeric outliers (IQR)", value=False, key="cap_outliers")
            drop_constant = st.checkbox("Drop constant columns", value=True, key="drop_constant")

        with st.spinner("Applying cleaning actions..."):
            cleaned_df = apply_cleaning_actions(
                df,
                remove_duplicates=remove_duplicates,
                impute_missing=impute_missing,
                cap_outliers=cap_outliers,
                drop_constant=drop_constant,
            )

        before_col, after_col = st.columns(2)
        with before_col:
            st.markdown("**Before**")
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", f"{df.shape[1]:,}")
        with after_col:
            st.markdown("**After**")
            st.metric("Rows", f"{len(cleaned_df):,}")
            st.metric("Columns", f"{cleaned_df.shape[1]:,}")

        st.caption("Cleaned data preview")
        st.dataframe(cleaned_df.head(15), use_container_width=True)

    analysis_df = cleaned_df

    # ============== DATA QUALITY SCORE ==============
    with st.expander("**4. Data Quality Score**", expanded=True):
        score, color = compute_quality_score(analysis_df)
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color: {color};">{score}</div>
            <div class="score-label">out of 100</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Based on: missing values, duplicate rows, and data type consistency.")

    # ============== MISSING VALUES ==============
    with st.expander("**5. Missing Values Analysis**", expanded=True):
        missing_counts = analysis_df.isna().sum()
        missing_pct = (missing_counts / len(analysis_df) * 100).round(1)
        missing_df = pd.DataFrame({
            "Column": missing_counts.index,
            "Missing count": missing_counts.values,
            "Missing %": missing_pct.values,
        })
        missing_df = missing_df[missing_df["Missing count"] > 0].sort_values("Missing %", ascending=False)

        if len(missing_df) == 0:
            st.success("No missing values found in your data.")
        else:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.4)))
            bars = ax.barh(missing_df["Column"], missing_df["Missing %"], color=ACCENT, alpha=0.8)
            ax.set_xlabel("Missing %")
            ax.set_title("Missing values by column")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    num_cols = get_numerical_columns(analysis_df)
    outlier_counts = {}
    strong_pairs = []

    # ============== OUTLIER DETECTION ==============
    with st.expander("**6. Outlier Detection (IQR)**", expanded=True):
        if not num_cols:
            st.info("No numerical columns found. Outlier detection applies only to numeric features.")
        else:
            outlier_counts = compute_outlier_counts_cached(analysis_df)
            outlier_series = pd.Series(outlier_counts)
            outlier_series = outlier_series[outlier_series > 0].sort_values(ascending=False)

            if len(outlier_series) == 0:
                st.success("No outliers detected in numerical columns (IQR method).")
            else:
                st.dataframe(
                    pd.DataFrame({"Column": outlier_series.index, "Outlier count": outlier_series.values}),
                    use_container_width=True,
                    hide_index=True,
                )
                fig, ax = plt.subplots(figsize=(10, max(4, len(outlier_series) * 0.4)))
                ax.barh(outlier_series.index, outlier_series.values, color=ACCENT, alpha=0.8)
                ax.set_xlabel("Number of outliers")
                ax.set_title("Outliers per column (1.5 × IQR)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # ============== FEATURE CORRELATION ==============
    with st.expander("**7. Feature Correlation**", expanded=True):
        num_cols = get_numerical_columns(analysis_df)
        if len(num_cols) < 2:
            st.info("Need at least 2 numerical columns for a correlation heatmap.")
        else:
            corr = analysis_df[num_cols].corr()
            if len(corr.columns) > 15:
                st.caption("Showing first 15 numerical columns for readability.")
                corr = corr.iloc[:15, :15]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                        ax=ax, linewidths=0.5, cbar_kws={"label": "Correlation"},
                        annot_kws={"color": TEXT, "size": 8})
            ax.set_title("Correlation heatmap (numerical features)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Strong pairs
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    v = corr.iloc[i, j]
                    if abs(v) >= 0.6:
                        strong_pairs.append((corr.columns[i], corr.columns[j], v))
            if strong_pairs:
                st.markdown("**Strongly correlated pairs (|r| ≥ 0.6):**")
                for a, b, r in strong_pairs[:10]:
                    st.write(f"- **{a}** and **{b}**: correlation = {r:.2f}. {'They move together.' if r > 0 else 'When one goes up, the other tends to go down.'}")
                st.caption("High correlation means one feature may be redundant with the other; consider dropping one if you want simpler models.")

    # ============== FEATURE IMPORTANCE ==============
    with st.expander("**8. Feature Importance (Random Forest)**", expanded=True):
        target_col = st.selectbox(
            "Select target column (what you want to predict)",
            options=analysis_df.columns.tolist(),
            index=0,
            key="target_col",
        )
        if target_col:
            with st.spinner("Computing feature importance..."):
                imp, perf_value, perf_metric = feature_importance_rf(analysis_df, target_col)
            if imp is None or len(imp) == 0:
                st.warning("Could not compute feature importance (e.g. not enough numerical features or valid target).")
            else:
                fig, ax = plt.subplots(figsize=(10, max(4, len(imp) * 0.35)))
                imp.plot(kind="barh", ax=ax, color=ACCENT, alpha=0.8)
                ax.set_xlabel("Importance")
                ax.set_title(f"Feature importance for predicting '{target_col}'")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                top3 = list(imp.tail(3).index)
                st.caption(f"**In plain English:** The most important features for predicting **{target_col}** are: **{', '.join(top3)}**. Focus on these when interpreting your model.")
                if perf_value is not None and perf_metric is not None:
                    st.metric(f"Holdout {perf_metric}", f"{perf_value:.3f}")

    # ============== RECOMMENDATIONS ==============
    with st.expander("**9. Plain English Recommendations**", expanded=True):
        missing_pct_series = (analysis_df.isna().sum() / len(analysis_df) * 100)
        recs = generate_recommendations(
            analysis_df, score,
            missing_pct_series,
            outlier_counts,
            len(strong_pairs) > 0,
        )
        for r in recs:
            st.markdown(f"- {r}")
        st.markdown('<div class="recommendations-box">**Tip:** Fix data quality issues before training models. Clean data leads to reliable results.</div>', unsafe_allow_html=True)

    with st.expander("**10. Export Cleaned Data + Report**", expanded=True):
        st.download_button(
            "Download cleaned CSV",
            analysis_df.to_csv(index=False).encode("utf-8"),
            "dataready_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )
        report_text = build_report_text(df, analysis_df, score, recs, strong_pairs)
        st.download_button(
            "Download analysis report (.txt)",
            report_text.encode("utf-8"),
            "dataready_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption("DataReady by ASX Labs — Clean Data. Better Models.")



if __name__ == "__main__":
    main()

