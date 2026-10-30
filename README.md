# 🏠 House Price Prediction
### End-to-End Regression Pipeline — Seattle Area, Washington

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![Sklearn](https://img.shields.io/badge/Scikit--learn-1.9-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## 📋 Project Overview

A complete **end-to-end regression pipeline** that predicts house sale prices
in the Seattle area of Washington State, using real estate sales from
May to July 2014.

This project demonstrates production-ready data science practices: leakage-free
pipelines, deliberate feature engineering, multi-model comparison, a log-price
target, model explanation with SHAP, and honest evaluation on unseen data.

It also includes a **Streamlit web app** where anyone can enter a house's details
and get a predicted price (see **Open the App** below).

---

## 🎯 Objective

Predict the **sale price** of a house given its physical characteristics,
location, and condition — using only information available at the time of listing.

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Source** | [USA House Prices — Kaggle (`fratzcan/usa-house-prices`)](https://www.kaggle.com/datasets/fratzcan/usa-house-prices) |
| **Size** | 4,140 house sales, 18 columns |
| **Period** | May — July 2014 |
| **Location** | Seattle area, Washington, USA (43 cities, 77 ZIP codes) |
| **Target** | `price` (continuous, USD) |
| **Features** | Bedrooms, bathrooms, sqft, floors, waterfront, view, condition, year built, renovations, city, ZIP code |

---

## 🗺️ Pipeline

| Step | Description |
|---|---|
| 1 | Load the data with `kagglehub` and inspect its structure |
| 2 | Structural cleaning — column sanitization, duplicates, missing-value check |
| 3 | Remove data-entry errors with fixed per-row rules — $0 prices and impossible price per sq ft (below $50 or above $1,500) |
| 4 | Feature engineering — `house_age`, `was_renovated`, `years_since_update` |
| 5 | Exploratory Data Analysis — Sweetviz report on the cleaned dataset |
| 6 | Train/test split (80/20) |
| 7 | Preprocessing pipeline — imputation + scaling, one-hot `city`, target-encoded `statezip` |
| 8 | All models trained on **log(price)** with `TransformedTargetRegressor` |
| 9 | Baseline model — `DummyRegressor` |
| 10 | Model comparison — Ridge, Random Forest, XGBoost, LightGBM (5-fold CV) |
| 11 | Hyperparameter tuning — `GridSearchCV` on the winner, optimizing CV MAE |
| 12 | Final evaluation — MAE, RMSE, R², MAPE and error analysis on the test set |
| 13 | Export — full pipeline saved with `joblib`, plus `locations.csv` for the app |
| 14 | Model explanation — SHAP |
| 15 | Streamlit web app — `app.py` |

Steps 3 and 4 run before the split because each rule or feature uses only the
house's own values. Everything **learned from the data** (medians, scaling,
target encoding) happens inside the pipeline and is fitted on training data only.

---

## 🔧 Feature Engineering

Three new features derived from raw columns:

| Feature | Description |
|---|---|
| `house_age` | How old the house was when sold (`sale_year - yr_built`) |
| `was_renovated` | Binary flag — was the house ever renovated? |
| `years_since_update` | Years since last renovation or original build |

Raw columns (`yr_built`, `yr_renovated`, `date`) are dropped after derivation, along with
`street` (almost unique per house) and `country` (always "USA").

---

## 🤖 Models Compared

5-fold cross-validation on the training set (3,268 houses):

| Model | CV MAE | CV RMSE | CV R² |
|---|---|---|---|
| **LightGBM 🏆 Winner** | **$92,761** | **$180,091** | **0.750** |
| Ridge | $95,218 | $190,483 | 0.721 |
| Random Forest | $96,061 | $181,349 | 0.746 |
| XGBoost | $97,085 | $192,917 | 0.713 |
| Baseline (DummyRegressor) | $228,721 | $367,641 | -0.048 |

**LightGBM won** with a CV MAE 59% lower than the baseline. All four models are
within about $4,300 of each other; Ridge coming second shows that most of the
price is explained by size, condition and location.

---

## 🔧 Hyperparameter Tuning

| Tool | Parameters tuned | Best parameters | Result |
|---|---|---|---|
| `GridSearchCV` (5-fold, CV MAE) | `n_estimators`, `learning_rate`, `num_leaves` | `n_estimators=300`, `learning_rate=0.06`, `num_leaves=7` | CV MAE $90,427 · CV RMSE $171,985 |

---

## 📊 Final Evaluation

Evaluated once on **817 unseen test houses**:

| Metric | Test Set |
|---|---|
| **MAE** | $93,890 |
| **RMSE** | $209,129 |
| **R²** | 0.732 |
| **MAPE** | 16.4% |

- 73% of predictions are within 20% of the real price, 44% within 10%.
- The test RMSE is 21.6% above the CV RMSE, but almost all of that gap comes from
  **one $7.06M house** predicted at $3.08M. Without it, the test RMSE is $156,133,
  below the CV RMSE.

---

## 🧠 Model Explanation (SHAP)

Typical size of each feature's effect on the predicted price:

| Feature | Typical effect |
|---|---|
| `statezip` (ZIP code) | ~30% |
| `sqft_living` | ~20% |
| `city` | ~8% |
| `sqft_above` | ~4.5% |
| `view` | ~3.3% |

**Location and size drive the price.** Age, renovation history and the number of
bedrooms matter much less.

---

## ⚠️ Key Findings & Limitations

**1. LightGBM beat all models, but narrowly**
All four models cut the baseline error by more than half and are close to each other.

**2. Very expensive homes are underestimated**
The model recognizes what makes a luxury house special (size, ZIP code, waterfront),
but the training data contains very few of them, so it can't learn how high such
prices go. Cheap houses, in contrast, tend to be predicted too high.

**3. Small dataset**
About 3,300 training houses, covering only three months of 2014.

**4. Feature engineering is outside the saved pipeline**
New houses must have `house_age`, `was_renovated` and `years_since_update`
calculated before calling `.predict()`. The Streamlit app does this for you;
in your own code, see the example below.

---

## 🖥️ Open the App

The Streamlit app (`app.py`) lets you enter a house's details and get a predicted price.

**What it does**
- Pick the **city**, then a **ZIP code** from that city (the list comes from `locations.csv`)
- Enter bedrooms, bathrooms, floors, area, lot size, year built and year renovated
- Set the view, the condition and whether the house is on the waterfront
- Click **Predict price**: the app calculates `house_age`, `was_renovated` and
  `years_since_update` exactly like the notebook, then shows the predicted price

Prices are in **2014 dollars**, because the model was trained on 2014 sales.

**How to open it**

1. Follow steps 1–3 of **How to Run** below (clone, virtual environment, install the requirements).
2. Install Streamlit, if it isn't installed yet:
   ```bash
   pip install streamlit
   ```
3. Make sure `house_price_model.pkl` and `locations.csv` are in the project folder.
   Both are in the repository; running the notebook creates them again.
4. Start the app from the project folder:
   ```bash
   streamlit run app.py
   ```
5. Your browser opens the app at **http://localhost:8501**. If it doesn't, open that address yourself.
6. To stop the app, press `Ctrl + C` in the terminal.

---

## 🚀 Next Steps

- Publish the app on Streamlit Community Cloud so it has a public link
- Show a SHAP chart in the app explaining each prediction
- Move the feature engineering into the pipeline so the model takes raw columns
- Add geospatial features — distance to city center, school district ratings
- Collect more data, especially for expensive homes
- Add a FastAPI endpoint for other programs to request predictions

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.13+ | Core language |
| Pandas | 3.1 | Data manipulation |
| Scikit-learn | 1.9 | Modeling & pipelines |
| XGBoost | 3.4 | Gradient boosting |
| LightGBM | 4.7 | Gradient boosting |
| SHAP | 0.54 | Model explanation |
| Sweetviz | 2.3 | EDA report |
| Matplotlib/Seaborn | — | Visualization |
| Joblib | — | Model export |
| Streamlit | — | Web app |

---

## 📁 Project Structure

```
House_Prices_Prediction/
│
├── project.ipynb              # Main notebook — full pipeline
├── app.py                     # Streamlit web app
├── house_price_model.pkl      # Exported pipeline (preprocessing + tuned LightGBM)
├── locations.csv              # Valid city / ZIP code pairs for the app's dropdowns
├── House_Prices_Report.html   # Sweetviz EDA report (cleaned dataset)
├── requirements.txt
├── LICENSE
└── README.md
```

The dataset is downloaded automatically by `kagglehub` when the notebook runs.

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/MaykInvest/House_Prices_Prediction.git
cd House_Prices_Prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter lab project.ipynb
```

---

## 🔮 Inference — Predict on New Data

```python
import joblib
import pandas as pd

# Load the exported pipeline
model = joblib.load('house_price_model.pkl')

# Predict on a new house (the three engineered features must be calculated first)
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

## 📝 Changelog

**Latest update**
- ✨ **Added a Streamlit web app** (`app.py`) to predict a house's price from a form
- ✨ Added `locations.csv` (valid city / ZIP code pairs), exported by the notebook for the app
- 🔧 Hyperparameter tuning now optimizes **CV MAE**, the same metric used to choose the winning model
- 🔧 The test RMSE without the most expensive house is now calculated in the notebook instead of written by hand
- 🔧 SHAP: the one-hot city effects are added up with their signs before measuring their size
- 📝 Notebook text fixed to match the code (when the cleaning runs, old tuning numbers, typos)
- 📝 Notebook re-run from top to bottom in a fresh kernel
- 📝 README rewritten to match the current project
- 🛠️ `requirements.txt` converted from UTF-16 to UTF-8 so `pip install -r` works

---

## 👤 Author

**Maykon Jonattan**
---

*This project was built as part of a data science portfolio demonstrating
end-to-end regression modeling with production-ready practices.*
