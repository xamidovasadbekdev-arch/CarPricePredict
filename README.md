# Car Price Prediction 🚗

A machine learning project that predicts used car selling prices based on vehicle specifications and history.

**Dataset:** [Vehicle Dataset from CarDekho — Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

---

## Problem Statement

Used car pricing is inconsistent and often opaque. This project builds a regression model that predicts a car's fair market price based on its specifications, helping both buyers and sellers make informed decisions.

---

## Project Structure

```
CarPricePrediction/
├── data/
│   ├── raw/          ← original Kaggle CSV (not committed)
│   └── processed/    ← cleaned dataset
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_FeatureEngineering.ipynb
│   └── 03_Modelling.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── models/           ← saved pipeline (not committed)
├── api/
│   └── main.py
├── tests/
├── .gitignore
├── requirements.txt
└── Dockerfile
```

---

## Dataset

- **Source:** CarDekho — India's leading used car platform
- **Size:** 6,926 cars after cleaning
- **Features:** year, km_driven, fuel type, seller type, transmission, owner history, mileage, engine, max_power, seats
- **Target:** `selling_price` (Indian Rupees)

To run this project, download the dataset from the Kaggle link above and place it at `data/raw/cardekho.csv`.

---

## Workflow

### 1. Data Cleaning
- Fixed column name formatting
- Removed duplicate entries
- Identified null values across all columns (~3% missing per column) — handled inside sklearn Pipeline to prevent data leakage

### 2. Exploratory Data Analysis

Key findings from EDA:

- **Price distribution:** Majority of cars sell between ₹100,000 and ₹2,000,000 with a right skew (skewness = 5.57) — log transformation applied to target
- **km_driven:** Negative relationship with price — higher mileage cars sell for less
- **Year:** Strong positive relationship — newer cars command higher prices
- **max_power:** Strongest numerical predictor (correlation = 0.69 with selling price)
- **engine:** Second strongest predictor (correlation = 0.44)
- **Seller type:** 85–90% of cars sold by individual owners (6,218 out of 6,926)
- **Fuel type:** Diesel (3,755) and Petrol (3,077) dominate the market
- **Transmission:** 91.5% manual — strong class imbalance in this feature
- **Seats insight:** 76% of cars have 5 seats (max price ₹470,000), but the most expensive cars have 7 seats (14% of data). Only 2 cars had 2 seats, both selling around ₹700,000–₹800,000

**Correlation heatmap highlights:**

| Feature | Correlation with Price |
|---|---|
| max_power | 0.69 |
| engine | 0.44 |
| year | 0.43 |
| seats | 0.16 |
| km_driven | -0.17 |
| mileage | -0.11 |

### 3. Feature Engineering

- **Brand extracted** from full car name using string split (`"Maruti Suzuki Swift"` → `"Maruti"`)
- **`car_age` created** from manufacturing year (`current_year - year`), more meaningful than raw year
- **`max_power` dtype fixed** from object to float (had mixed string/numeric values)
- **`mileage` column name** standardized
- **`name` and `year` columns dropped** after extraction

### 4. Modelling

**Preprocessing pipeline** built with `ColumnTransformer` to prevent data leakage — all transformers fit only on training data:

- **Numerical features** (`km_driven`, `mileage`, `engine`, `max_power`, `seats`, `car_age`): `SimpleImputer(median)` → `StandardScaler`
- **Categorical features** (`fuel`, `seller_type`, `transmission`, `brand`): `SimpleImputer(most_frequent)` → `OneHotEncoder`
- **`owner`** (ordinal — has real order: First > Second > Third > Fourth): `SimpleImputer(most_frequent)` → `OrdinalEncoder`
- **Target transform:** `selling_price` was right-skewed (skewness = 5.57), so `np.log1p()` applied to target before training, reversed with `np.expm1()` at prediction time — skewness after transform dropped to -0.15

**Train/test split:** 80/20

**Models compared:**

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.8671 | ₹91,413 | ₹170,752 |
| **Random Forest** | **0.9245** | **₹73,340** | **₹128,706** |
| XGBoost | 0.9201 | ₹72,095 | ₹132,338 |

**Random Forest selected as final model** — best overall balance of R² and RMSE.

**Final model performance (on test set, real ₹ scale):**

| Metric | Score |
|---|---|
| R² | 0.9245 |
| MAE | ₹73,340 |
| RMSE | ₹128,706 |

### 5. Deployment

- Model saved as a complete sklearn Pipeline (preprocessing + model together) using `joblib` — ensures raw, unprocessed input can be fed directly to the API without manual scaling/encoding
- Wrapped in a **FastAPI** REST API
- Containerized with **Docker**
- Deployed on **Render**

---

## Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn
- **Visualization** — Matplotlib, Seaborn
- **API** — FastAPI, Pydantic
- **Deployment** — Docker, Render

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/xamidovasadbekdev-arch/CarPricePredict.git
cd CarPricePredict

# Create environment
conda create --prefix ./env python=3.11
conda activate ./env

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place at:
# data/raw/cardekho.csv

# Run notebooks in order:
# 01_EDA.ipynb → 02_FeatureEngineering.ipynb → 03_Modelling.ipynb
```

---

## API Usage

```bash
POST /predict
Content-Type: application/json

{
  "km_driven": 45000,
  "fuel": "Petrol",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": "First Owner",
  "mileage": 18.5,
  "engine": 1197.0,
  "max_power": 82.0,
  "seats": 5.0,
  "brand": "Maruti",
  "car_age": 5
}
```

Response:
```json
{
  "predicted_price": 517823
}
```

---

## Key Takeaways

- Built a fully leak-free pipeline — all preprocessing (imputation, scaling, encoding) fit exclusively on training data
- Compared three regression algorithms and selected the best based on test set performance, not just training accuracy
- Validated stability with cross-validation before trusting the final metrics
- Learned to reverse a target transformation (log1p/expm1) correctly when evaluating in real-world units
- Made a data-driven decision to reject hyperparameter tuning results when they degraded test performance, rather than blindly trusting CV scores

---

## Author

**Asadbek** — Junior ML Engineer  
[GitHub](https://github.com/xamidovasadbekdev-arch) • [LinkedIn](https://www.linkedin.com/in/asadbekxamidov/)
