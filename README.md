# 🏠 House Price Prediction
### End-to-End Regression Pipeline — King County, Washington

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Sklearn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## 📋 Project Overview

A complete **end-to-end regression pipeline** that predicts house sale prices
in King County, Washington — including Seattle — using real estate transaction
data from May 2014 to May 2015.

This project demonstrates production-ready data science practices: leakage-free
pipelines, deliberate feature engineering, multi-model comparison, and honest
evaluation on completely unseen data.

---

## 🎯 Objective

Predict the **sale price** of a house given its physical characteristics,
location, and condition — using only information available at the time of listing.

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Source** | King County House Sales — Kaggle |
| **Size** | ~21,000 house sales |
| **Period** | May 2014 — May 2015 |
| **Location** | King County, Washington, USA |
| **Target** | `price` (continuous, USD) |
| **Features** | Bedrooms, bathrooms, sqft, floors, waterfront, condition, location, year built, renovations |

---

## 🗺️ Pipeline

| Step | Description |
|---|---|
| 1 | Load & defensive copy (`df → dfc`) |
| 2 | Structural cleaning — duplicates, column sanitization |
| 3 | Exploratory Data Analysis (EDA) — Sweetviz report |
| 4 | Train/test split (80/20) |
| 5 | Outlier removal — IQR on `y_train` only |
| 6 | Feature engineering — `house_age`, `was_renovated`, `years_since_update` |
| 7 | Preprocessing pipeline — imputation + scaling + encoding |
| 8 | Baseline model — `DummyRegressor` |
| 9 | Model comparison — Ridge, Random Forest, XGBoost, LightGBM |
| 10 | Hyperparameter tuning — `BayesSearchCV` on the winner |
| 11 | Final evaluation — MAE, RMSE, R², dual test set analysis |
| 12 | Export — full pipeline saved with `joblib` |

---

## 🔧 Feature Engineering

Three new features derived from raw columns:

| Feature | Description |
|---|---|
| `house_age` | How old the house was when sold (`sale_year - yr_built`) |
| `was_renovated` | Binary flag — was the house ever renovated? |
| `years_since_update` | Years since last renovation or original build |

Raw columns (`yr_built`, `yr_renovated`, `date`) dropped after derivation.

---

## 🤖 Models Compared

| Model | CV MAE | CV RMSE |
|---|---|---|
| Baseline (DummyRegressor) | $175,398 | $217,153 |
| **LightGBM 🏆 Winner** | **$85,003** | **$119,768** |
| Ridge | $85,565 | $120,270 |
| XGBoost | $87,044 | $124,121 |
| Random Forest | $87,643 | $124,121 |

**LightGBM won** — the gradient boosting model edged out Ridge by a small
margin on both MAE and RMSE. The close competition between LightGBM and Ridge
suggests the feature-price relationship has both linear and non-linear components.

---

## 🔧 Hyperparameter Tuning

| Tool | Parameters tuned | Best result |
|---|---|---|
| `GridSearchCV` | `n_estimators`, `max_depth`, `learning_rate`, `num_leaves` | CV RMSE = $119,242 |

Key parameters found: `learning_rate=0.2`, `max_depth=5`, `n_estimators=100`, `num_leaves=7`.

---

## 📊 Final Evaluation

Evaluated on **828 completely unseen test houses**:

| Metric | Full Test Set | Valid Prices Only |
|---|---|---|
| **MAE** | $122,385 | $116,749 |
| **RMSE** | $231,669 | $221,645 |
| **R²** | 0.4880 | 0.5209 |

> ⚠️ **Note:** 10 houses in the test set had `price = $0` — likely data entry
> errors. Results are reported for both the full test set and valid-price-only
> subset for transparency.

---

## ⚠️ Key Findings & Limitations

**1. LightGBM beat all models**
The gradient boosting model narrowly outperformed Ridge, XGBoost, and Random
Forest — suggesting the feature-price relationship has both linear and
non-linear components that LightGBM captures better.

**2. Large CV vs. test gap — data distribution mismatch**
Training CV RMSE = $119,242 vs Test RMSE = $221,645 (85.9% gap). This reflects
a structural limitation — luxury homes above $1.5M were removed from training
via IQR outlier filtering but remain in the test set. The model was never exposed
to that price range and predictably underestimates them. This is a data
distribution mismatch, not a pipeline error.

**3. R² of 0.52 — room for improvement**
The model explains ~52% of price variance. The remaining 48% likely comes from
hyperlocal factors (school district, neighborhood desirability, view) not fully
captured in the available features.

---

## 🚀 Next Steps

- Build a separate model for luxury homes (above $1.5M) to close the CV/test gap
- Add geospatial features — distance to city center, school district ratings
- Test target encoding on `statezip` to recover granularity without one-hot sparsity
- Collect more data — ~3,300 training rows limits what LightGBM can learn
- Deploy as a FastAPI endpoint for real-time price predictions

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Pandas | 2.0+ | Data manipulation |
| Scikit-learn | 1.3+ | Modeling & pipelines |
| XGBoost | 1.7+ | Gradient boosting |
| LightGBM | 3.3+ | Gradient boosting |
| Scikit-Optimize | 0.9+ | Bayesian hypertuning |
| Sweetviz | 2.1+ | EDA report |
| Matplotlib/Seaborn | — | Visualization |
| Joblib | — | Model export |

---

## 📁 Project Structure

```
house_price_prediction/
│
├── data/                    # Raw and cleaned datasets
├── notebooks/
│   └── project.ipynb        # Main notebook — full pipeline
├── models/
│   └── house_price_pipeline.pkl  # Exported LightGBM pipeline
├── reports/
│   └── House_Price_Report.html   # Sweetviz EDA report
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook notebooks/project.ipynb
```

---

## 🔮 Inference — Predict on New Data

```python
import joblib
import pandas as pd

# Load the exported pipeline
model = joblib.load('models/house_price_pipeline.pkl')

# Predict on a new house
new_house = pd.DataFrame([{
    'bedrooms'        : 3,
    'bathrooms'       : 2.0,
    'sqft_living'     : 1800,
    'sqft_lot'        : 5000,
    'floors'          : 1.0,
    'waterfront'      : 0,
    'view'            : 0,
    'condition'       : 3,
    'sqft_above'      : 1800,
    'sqft_basement'   : 0,
    'house_age'       : 25,
    'was_renovated'   : 0,
    'years_since_update': 25,
    'city'            : 'Seattle',
    'statezip'        : 'WA 98101'
}])

predicted_price = model.predict(new_house)[0]
print(f"Predicted price: ${predicted_price:,.0f}")
```

---

## 👤 Author

**Maykon Jonattan**
---

*This project was built as part of a data science portfolio demonstrating
end-to-end regression modeling with production-ready practices.*