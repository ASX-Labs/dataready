# DataReady – Updated Features & Overview

🚀 **Live Demo**
https://dataready-jntbkbxpzsudwabniq4imq.streamlit.app

**Professional data analysis web app by ASX Labs — Clean Data. Better Models.**

---

## 🧠 Overview

DataReady is an interactive data analysis and preprocessing tool designed to help users quickly understand, clean, and prepare datasets for machine learning. It combines automated data profiling, visualization, and basic modeling into a clean, user-friendly interface.

---

## ⚡ Quick Start

```bash
cd DataReady
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in your terminal.

---

## 🚀 Core Features

### 📂 CSV Upload

* Drag & drop CSV upload
* Automatic delimiter detection
* Displays:

  * Row count
  * Column count
  * File size

---

### 📊 Data Quality Score

* Score from **0–100**
* Based on:

  * Missing values
  * Duplicate rows
  * Data type consistency

---

### 🧩 Data Type Profiler

* Column-level insights:

  * Pandas dtype
  * Inferred type (numeric, categorical, datetime)
  * Missing %
  * Unique values
  * Flags:

    * Possible ID columns
    * High missingness
    * Constant columns

---

### 🧹 Data Cleaning (Interactive)

* Toggle-based cleaning:

  * Remove duplicates
  * Impute missing values (median/mode)
  * Cap outliers (IQR method)
  * Drop constant columns

#### ✨ Auto Clean (New)

* One-click recommended cleaning setup

#### 👀 Preview

* Before vs After comparison
* Cleaned dataset preview table

---

### ⚠️ Missing Values Analysis

* Per-column:

  * Missing count
  * Missing %
* Horizontal bar chart visualization

---

### 📈 Outlier Detection (IQR)

* Uses **1.5 × IQR rule**
* Displays:

  * Outlier counts per column
  * Visualization chart

---

### 🔗 Feature Correlation

* Correlation heatmap (numeric features)
* Automatically detects **strong correlations (|r| ≥ 0.6)**
* Plain-English explanations:

  * Positive vs negative relationships

---

### 🤖 Feature Importance (Enhanced)

* Uses **Random Forest models**
* Supports:

  * Classification (Accuracy)
  * Regression (R² Score)

#### Includes:

* Train/test split (holdout validation)
* Performance metric display
* Ranked feature importance chart

---

### 💡 Recommendations Engine

* Generates **plain English suggestions**
* Covers:

  * Missing data handling
  * Duplicate removal
  * Outlier treatment
  * Feature correlation issues

---

## 🔄 Important Behavior

All analytics run on the **cleaned dataset**, not the raw upload.

This affects:

* Data Quality Score
* Missing Values
* Outlier Detection
* Correlation
* Feature Importance
* Recommendations

---

## 📤 Export Options

### 📁 Cleaned Dataset

* Download as CSV

### 📝 Analysis Report

* Includes:

  * Dataset overview
  * Quality score
  * Duplicate count
  * Top missing issues
  * Correlation insights
  * Recommendations

---

## 🛠 Tech Stack

* Python
* Streamlit
* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn

---

## 🎨 Branding

* Dark theme: `#000000`
* Accent color: `#2ECC71`
* Clean UI with ASX Labs branding

---

## 🧪 Positioning

DataReady combines:

* Data profiling
* Automated cleaning
* Basic machine learning insights

All in one lightweight web application.

---

**DataReady by ASX Labs — Clean Data. Better Models.**
