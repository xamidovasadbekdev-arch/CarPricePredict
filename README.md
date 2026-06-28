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


## Author

**Asadbek** — Junior ML Engineer  
[GitHub](https://github.com/xamidovasadbekdev-arch) • [LinkedIn](https://www.linkedin.com/in/asadbekxamidov/)
